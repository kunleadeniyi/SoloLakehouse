"""Convention-based OpenMetadata lineage helper for Dagster assets.

FQN derivation — no hardcoded entity names:
  MinIO path   → {OM_MINIO_SERVICE}.{OM_MINIO_BUCKET}.{tier}/{name}
                 derived from first two components of the object path
  Trino table  → {OM_TRINO_SERVICE}.{catalog}.{schema}.{table}
  MLflow model → {OM_MLFLOW_SERVICE}.{model_name}

Adding a new data source requires no changes here: as long as the MinIO
container is registered in OM and the asset returns its MinIO path, lineage
is derived automatically.

Silently no-ops when OM_JWT_TOKEN is unset so the pipeline runs without OM.
"""
from __future__ import annotations

import os
from urllib.parse import quote

import requests
import structlog

logger = structlog.get_logger()

_OM_URL = os.environ.get("OM_SERVER_URL", "http://openmetadata-server:8585")
_OM_TOKEN = os.environ.get("OM_JWT_TOKEN", "")
_MINIO_SERVICE = os.environ.get("OM_MINIO_SERVICE", "slh-minio")
_MINIO_BUCKET = os.environ.get("OM_MINIO_BUCKET", "sololakehouse")
_TRINO_SERVICE = os.environ.get("OM_TRINO_SERVICE", "slh-trino")
_MLFLOW_SERVICE = os.environ.get("OM_MLFLOW_SERVICE", "slh-mlflow")

_ENDPOINTS: dict[str, str] = {
    "container": "containers",
    "table": "tables",
    "mlmodel": "mlmodels",
}


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_OM_TOKEN}", "Content-Type": "application/json"}


def _resolve_id(fqn: str, entity_type: str) -> str | None:
    endpoint = _ENDPOINTS.get(entity_type)
    if not endpoint:
        logger.warning("om_lineage_unknown_type", entity_type=entity_type)
        return None
    url = f"{_OM_URL}/api/v1/{endpoint}/name/{quote(fqn, safe='')}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()["id"]
        logger.warning("om_lineage_resolve_failed", fqn=fqn, status=resp.status_code)
    except Exception as exc:
        logger.warning("om_lineage_resolve_error", fqn=fqn, error=str(exc))
    return None


def add_lineage(from_fqn: str, from_type: str, to_fqn: str, to_type: str) -> None:
    """Create a lineage edge in OpenMetadata. No-op if OM_JWT_TOKEN is not configured."""
    if not _OM_TOKEN:
        return
    from_id = _resolve_id(from_fqn, from_type)
    to_id = _resolve_id(to_fqn, to_type)
    if not from_id or not to_id:
        return
    try:
        resp = requests.put(
            f"{_OM_URL}/api/v1/lineage",
            headers=_headers(),
            json={
                "edge": {
                    "fromEntity": {"id": from_id, "type": from_type},
                    "toEntity": {"id": to_id, "type": to_type},
                }
            },
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("om_lineage_added", from_fqn=from_fqn, to_fqn=to_fqn)
        else:
            logger.warning(
                "om_lineage_put_failed",
                from_fqn=from_fqn,
                to_fqn=to_fqn,
                status=resp.status_code,
            )
    except Exception as exc:
        logger.warning("om_lineage_error", from_fqn=from_fqn, to_fqn=to_fqn, error=str(exc))


# ---------------------------------------------------------------------------
# Convention-based FQN builders
# ---------------------------------------------------------------------------

def _container_fqn(minio_path: str) -> str:
    """Derive OM container FQN from a MinIO object path.

    "bronze/ecb_rates/ingestion_date=2026-05-08/ecb_rates.parquet"
        → "slh-minio.sololakehouse.bronze/ecb_rates"

    Works for any depth — only the first two path components are used,
    so new data sources wire up automatically without changing this file.
    """
    parts = minio_path.lstrip("/").split("/")
    container_path = f"{parts[0]}/{parts[1]}"
    return f"{_MINIO_SERVICE}.{_MINIO_BUCKET}.{container_path}"


def _table_fqn(catalog: str, schema: str, table: str) -> str:
    return f"{_TRINO_SERVICE}.{catalog}.{schema}.{table}"


def _mlmodel_fqn(model_name: str) -> str:
    return f"{_MLFLOW_SERVICE}.{model_name}"


# ---------------------------------------------------------------------------
# Typed convenience helpers — called from assets.py
# ---------------------------------------------------------------------------

def lineage_minio_to_minio(from_path: str, to_path: str) -> None:
    """Link two MinIO containers derived from their object paths."""
    add_lineage(_container_fqn(from_path), "container", _container_fqn(to_path), "container")


def lineage_minio_to_table(from_path: str, catalog: str, schema: str, table: str) -> None:
    """Link a MinIO container to a Trino table."""
    add_lineage(_container_fqn(from_path), "container", _table_fqn(catalog, schema, table), "table")


def lineage_table_to_mlmodel(catalog: str, schema: str, table: str, model_name: str) -> None:
    """Link a Trino table to an MLflow registered model."""
    add_lineage(_table_fqn(catalog, schema, table), "table", _mlmodel_fqn(model_name), "mlmodel")

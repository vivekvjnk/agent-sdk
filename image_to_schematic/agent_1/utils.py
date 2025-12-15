from typing import Optional
from google.cloud import storage

# ---------------------------------------------------------------------------
# Helper: Resolve GCS image URIs (read-only)
# ---------------------------------------------------------------------------

def find_segment_image_gcs_uri(
    bucket: storage.Bucket,
    bucket_name: str,
    images_prefix: str,
    image_stem: str,
) -> Optional[str]:
    """
    Given an image stem (e.g., 'bq79616_subckt_003_s40'), try to find a matching
    image blob in GCS under images_prefix with extensions .png/.jpg/.jpeg.
    Returns the gs:// URI if found, else None.
    """
    images_prefix = images_prefix.rstrip("/")
    for ext in [".png", ".jpg", ".jpeg"]:
        blob_name = f"{images_prefix}/{image_stem}{ext}"
        blob = bucket.blob(blob_name)
        if blob.exists():
            return f"gs://{bucket_name}/{blob_name}"
    return None

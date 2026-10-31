"""P3 Task 1 — Incident and Evidence Store.

Manages incident lifecycle, secure file uploads, SHA-256 content verification,
metadata tracking, and local storage references for multi-source forensic evidence:
- Dashcam / CCTV video footage (.mp4, .avi, .mov)
- Written witness statements (.txt, .json)
- Telematics / CAN bus / accelerometer sensor records (.csv)
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
import uuid


class EvidenceStore:
    """Manages forensic evidence files and incident metadata."""

    def __init__(self, base_dir: Path | str = "data/incidents") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Compute SHA-256 cryptographic hash of byte content."""
        return hashlib.sha256(content).hexdigest()

    def _incident_dir(self, incident_id: str) -> Path:
        inc_dir = self.base_dir / incident_id
        inc_dir.mkdir(parents=True, exist_ok=True)
        return inc_dir

    def _incident_file(self, incident_id: str) -> Path:
        return self._incident_dir(incident_id) / "incident.json"

    def create_incident(
        self,
        incident_id: Optional[str] = None,
        title: str = "Traffic Incident Investigation",
        description: str = "",
        location: str = "Intersection",
        incident_date: Optional[str] = None,
        created_by: str = "forensic_analyst",
    ) -> dict[str, Any]:
        """Create a new forensic incident record."""
        inc_id = incident_id or f"INC-{uuid.uuid4().hex[:8].upper()}"
        inc_file = self._incident_file(inc_id)

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "incident_id": inc_id,
            "title": title,
            "description": description,
            "location": location,
            "incident_date": incident_date or now_iso,
            "created_at": now_iso,
            "updated_at": now_iso,
            "created_by": created_by,
            "status": "created",
            "time_zero_offset_s": 0.0,
            "evidence_count": 0,
            "evidence": [],
        }

        inc_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def get_incident(self, incident_id: str) -> Optional[dict[str, Any]]:
        """Retrieve incident metadata."""
        inc_file = self._incident_file(incident_id)
        if not inc_file.exists():
            return None
        try:
            return json.loads(inc_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_incidents(self) -> list[dict[str, Any]]:
        """List all registered incidents."""
        results: list[dict[str, Any]] = []
        for inc_dir in sorted(self.base_dir.iterdir()):
            if inc_dir.is_dir():
                inc_file = inc_dir / "incident.json"
                if inc_file.exists():
                    try:
                        data = json.loads(inc_file.read_text(encoding="utf-8"))
                        results.append(data)
                    except Exception:
                        continue
        return results

    def store_evidence(
        self,
        incident_id: str,
        filename: str,
        content: bytes,
        evidence_type: str,  # "video" | "witness" | "sensor"
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Store an uploaded evidence file, compute SHA-256, and register it."""
        inc = self.get_incident(incident_id)
        if inc is None:
            inc = self.create_incident(incident_id=incident_id)

        evidence_dir = self._incident_dir(incident_id) / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        sha256 = self.compute_sha256(content)
        file_ext = Path(filename).suffix.lower()
        file_size = len(content)

        # Unique evidence ID
        prefix_map = {"video": "EVD-VID", "witness": "EVD-WIT", "sensor": "EVD-SEN"}
        type_prefix = prefix_map.get(evidence_type.lower(), "EVD-OTH")
        evd_id = f"{type_prefix}-{sha256[:8].upper()}"

        safe_filename = f"{evd_id}_{Path(filename).name}"
        dest_path = evidence_dir / safe_filename
        dest_path.write_bytes(content)

        now_iso = datetime.now(timezone.utc).isoformat()
        evidence_entry: dict[str, Any] = {
            "evidence_id": evd_id,
            "filename": filename,
            "stored_filename": safe_filename,
            "file_path": str(dest_path.resolve()),
            "relative_path": f"evidence/{safe_filename}",
            "evidence_type": evidence_type.lower(),
            "sha256": sha256,
            "size_bytes": file_size,
            "description": description,
            "uploaded_at": now_iso,
            "metadata": metadata or {},
        }

        # Update incident record
        # Remove if evidence with same sha256 already existed
        inc["evidence"] = [e for e in inc.get("evidence", []) if e.get("sha256") != sha256]
        inc["evidence"].append(evidence_entry)
        inc["evidence_count"] = len(inc["evidence"])
        inc["updated_at"] = now_iso
        inc["status"] = "evidence_uploaded"

        self._incident_file(incident_id).write_text(json.dumps(inc, indent=2), encoding="utf-8")
        return evidence_entry

    def list_evidence(self, incident_id: str) -> list[dict[str, Any]]:
        """List all evidence files registered for an incident."""
        inc = self.get_incident(incident_id)
        if inc is None:
            return []
        return inc.get("evidence", [])

    def get_evidence_file_path(self, incident_id: str, evidence_id: str) -> Optional[Path]:
        """Get absolute path to a specific stored evidence file."""
        inc = self.get_incident(incident_id)
        if not inc:
            return None
        for ev in inc.get("evidence", []):
            if ev.get("evidence_id") == evidence_id:
                p = Path(ev["file_path"])
                if p.exists():
                    return p
        return None

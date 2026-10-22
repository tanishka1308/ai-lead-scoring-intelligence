"""Validation module for incoming lead records."""
import math
from typing import Any, Dict, List, Tuple
from lead_scoring.config import (
    ALLOWED_INDUSTRIES,
    ALLOWED_JOB_LEVELS,
    ALLOWED_LEAD_SOURCES,
    REQUIRED_FIELDS,
)


def _is_missing(val: Any) -> bool:
    """Check if value is None, blank string, or NaN."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _parse_bool_demo(val: Any) -> Tuple[int | None, bool]:
    """Parse demo_requested to 0 or 1. Returns (parsed_val, is_valid)."""
    if _is_missing(val):
        return None, False
    if isinstance(val, bool):
        return int(val), True
    if isinstance(val, (int, float)):
        if val in (0, 1, 0.0, 1.0):
            return int(val), True
        return None, False
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "true", "yes", "y"):
            return 1, True
        if v in ("0", "false", "no", "n"):
            return 0, True
    return None, False


def _parse_int(val: Any, field_name: str, min_val: int = 0) -> Tuple[int | None, str | None]:
    """Parse integer field with minimum value check."""
    if _is_missing(val):
        return None, None
    try:
        f = float(val)
        if math.isnan(f) or not f.is_integer():
            return None, f"Field '{field_name}' must be an integer (received '{val}')."
        i = int(f)
        if i < min_val:
            return None, f"Field '{field_name}' must be >= {min_val} (received '{val}')."
        return i, None
    except (ValueError, TypeError):
        return None, f"Field '{field_name}' must be an integer (received '{val}')."


def _parse_float(val: Any, field_name: str, min_val: float = 0.0) -> Tuple[float | None, str | None]:
    """Parse float field with minimum value check."""
    if _is_missing(val):
        return None, None
    try:
        f = float(val)
        if math.isnan(f):
            return None, None
        if f < min_val:
            return None, f"Field '{field_name}' must be a number >= {min_val} (received '{val}')."
        return f, None
    except (ValueError, TypeError):
        return None, f"Field '{field_name}' must be a numeric value (received '{val}')."


def validate_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a single lead record.

    Args:
        record: Dictionary representing a raw or incoming lead.

    Returns:
        Tuple of (clean_dict, errors_list).
        If errors_list is non-empty, the record is invalid.
    """
    errors: List[str] = []
    clean: Dict[str, Any] = {}

    # lead_id (optional, passed through or stripped)
    raw_lead_id = record.get("lead_id")
    if not _is_missing(raw_lead_id):
        clean["lead_id"] = str(raw_lead_id).strip()
    else:
        clean["lead_id"] = None

    # Check required fields presence
    for req in REQUIRED_FIELDS:
        val = record.get(req)
        if _is_missing(val):
            errors.append(f"Field '{req}' is required and cannot be empty.")

    # 1. lead_source
    raw_source = record.get("lead_source")
    if not _is_missing(raw_source):
        source_clean = str(raw_source).strip().lower()
        if source_clean in ALLOWED_LEAD_SOURCES:
            clean["lead_source"] = source_clean
        else:
            errors.append(
                f"Field 'lead_source' has invalid value '{raw_source}'. "
                f"Allowed values: {sorted(list(ALLOWED_LEAD_SOURCES))}."
            )
            clean["lead_source"] = source_clean
    else:
        clean["lead_source"] = None

    # 2. industry
    raw_ind = record.get("industry")
    if not _is_missing(raw_ind):
        ind_clean = str(raw_ind).strip().lower()
        if ind_clean in ALLOWED_INDUSTRIES:
            clean["industry"] = ind_clean
        else:
            errors.append(
                f"Field 'industry' has invalid value '{raw_ind}'. "
                f"Allowed values: {sorted(list(ALLOWED_INDUSTRIES))}."
            )
            clean["industry"] = ind_clean
    else:
        clean["industry"] = None

    # 3. job_level
    raw_job = record.get("job_level")
    if not _is_missing(raw_job):
        job_clean = str(raw_job).strip().lower()
        if job_clean in ALLOWED_JOB_LEVELS:
            clean["job_level"] = job_clean
        else:
            errors.append(
                f"Field 'job_level' has invalid value '{raw_job}'. "
                f"Allowed values: {sorted(list(ALLOWED_JOB_LEVELS))}."
            )
            clean["job_level"] = job_clean
    else:
        clean["job_level"] = None

    # 4. company_size (required integer >= 1)
    raw_cs = record.get("company_size")
    if not _is_missing(raw_cs):
        cs_val, cs_err = _parse_int(raw_cs, "company_size", min_val=1)
        if cs_err:
            errors.append(cs_err)
        clean["company_size"] = cs_val
    else:
        clean["company_size"] = None

    # 5. website_visits (required integer >= 0)
    raw_wv = record.get("website_visits")
    if not _is_missing(raw_wv):
        wv_val, wv_err = _parse_int(raw_wv, "website_visits", min_val=0)
        if wv_err:
            errors.append(wv_err)
        clean["website_visits"] = wv_val
    else:
        clean["website_visits"] = None

    # 6. demo_requested (required boolean / 0 or 1)
    raw_dr = record.get("demo_requested")
    if not _is_missing(raw_dr):
        dr_val, is_valid = _parse_bool_demo(raw_dr)
        if not is_valid:
            errors.append(
                f"Field 'demo_requested' must be boolean or 0/1/yes/no (received '{raw_dr}')."
            )
        clean["demo_requested"] = dr_val
    else:
        clean["demo_requested"] = None

    # 7. pages_per_visit (optional float >= 0)
    raw_ppv = record.get("pages_per_visit")
    ppv_val, ppv_err = _parse_float(raw_ppv, "pages_per_visit", min_val=0.0)
    if ppv_err:
        errors.append(ppv_err)
    clean["pages_per_visit"] = ppv_val

    # 8. emails_opened (optional int >= 0)
    raw_eo = record.get("emails_opened")
    eo_val, eo_err = _parse_int(raw_eo, "emails_opened", min_val=0)
    if eo_err:
        errors.append(eo_err)
    clean["emails_opened"] = eo_val

    # 9. email_clicks (optional int >= 0)
    raw_ec = record.get("email_clicks")
    ec_val, ec_err = _parse_int(raw_ec, "email_clicks", min_val=0)
    if ec_err:
        errors.append(ec_err)
    clean["email_clicks"] = ec_val

    # Cross-field rule: email_clicks <= emails_opened
    if clean["email_clicks"] is not None and clean["emails_opened"] is not None:
        if clean["email_clicks"] > clean["emails_opened"]:
            errors.append(
                f"Cross-field constraint violated: 'email_clicks' ({clean['email_clicks']}) "
                f"cannot exceed 'emails_opened' ({clean['emails_opened']})."
            )

    # 10. days_since_last_contact (optional float >= 0)
    raw_recency = record.get("days_since_last_contact")
    rec_val, rec_err = _parse_float(raw_recency, "days_since_last_contact", min_val=0.0)
    if rec_err:
        errors.append(rec_err)
    clean["days_since_last_contact"] = rec_val

    # 11. budget_usd (optional float >= 0)
    raw_budget = record.get("budget_usd")
    b_val, b_err = _parse_float(raw_budget, "budget_usd", min_val=0.0)
    if b_err:
        errors.append(b_err)
    clean["budget_usd"] = b_val

    # 12. country (free text)
    raw_country = record.get("country")
    if not _is_missing(raw_country):
        clean["country"] = str(raw_country).strip()
    else:
        clean["country"] = None

    return clean, errors

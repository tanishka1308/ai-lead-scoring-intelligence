from typing import List

from pydantic import BaseModel, ConfigDict, Field


class LeadFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_size: str = Field(min_length=1, max_length=40)
    industry: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=60)
    country: str = Field(min_length=2, max_length=60)
    days_since_contact: int = Field(ge=0, le=3650)
    website_visits: int = Field(ge=0, le=10000)
    emails_opened: int = Field(ge=0, le=10000)
    meeting_attended: bool


class BatchRequest(BaseModel):
    leads: List[LeadFeatures] = Field(min_length=1, max_length=500)

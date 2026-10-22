"""Create a deterministic, fictional lead dataset for demonstration."""
import csv
import random
from pathlib import Path

random.seed(2026)
OUT = Path(__file__).parent / "data" / "leads.csv"
OUT.parent.mkdir(exist_ok=True)

COMPANY = ["1-50", "51-200", "201-1000", "1000+"]
INDUSTRY = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
SOURCE = ["Webinar", "Referral", "Website", "Outbound", "Conference"]
COUNTRY = ["US", "IN", "GB", "CA", "AU"]
rows = []
for i in range(1200):
    size, industry, source, country = random.choice(COMPANY), random.choice(INDUSTRY), random.choice(SOURCE), random.choice(COUNTRY)
    days, visits, emails, meeting = random.randint(0, 120), random.randint(0, 30), random.randint(0, 15), random.random() < 0.38
    signal = -2.2 + visits * 0.10 + emails * 0.08 + meeting * 1.4 - days * 0.012
    signal += (size == "201-1000") * 0.25 + (size == "1000+") * 0.45
    signal += (source == "Referral") * 0.45 + (source == "Conference") * 0.2
    signal += (industry == "Technology") * 0.25 + (country == "US") * 0.1
    probability = 1 / (1 + pow(2.718281828, -signal))
    converted = int(random.random() < probability)
    rows.append({"lead_id": f"L{i+1:04d}", "company_size": size, "industry": industry, "source": source,
                 "country": country, "days_since_contact": days, "website_visits": visits,
                 "emails_opened": emails, "meeting_attended": meeting, "converted": converted})

# Add limited missing values and exact duplicates to demonstrate the cleaning workflow.
for i in range(0, len(rows), 97):
    rows[i]["industry"] = ""
for i in range(0, len(rows), 131):
    rows[i]["website_visits"] = ""
rows += [dict(row) for row in rows[:12]]
with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {OUT}")

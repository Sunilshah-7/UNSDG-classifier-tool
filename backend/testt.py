"""
test_ge_lab_alone.py

Calls the local GE-Lab microservice directly (no MiniLM, no ensemble)
to see how it scores known positive/negative projects in isolation.
Uses cached text — no re-fetching, no API credits spent.
"""

import requests
import sdg_constants

GE_LAB_URL = "http://127.0.0.1:9010/predict"

# ── Reuse the same cached text from your calibration run ─────────────────────
LABELED_PROJECTS = [
    {
        "name": "opentripplanner/OpenTripPlanner",
        "text": "Public transportation systems face challenges in providing efficient travel options, especially with disruptions and service changes. OpenTripPlanner addresses this issue by offering a multi-modal trip planner that combines scheduled public transportation with walking, bicycling, and mobility services. The project benefits communities by improving travel experiences and reducing reliance on personal vehicles. It operates in various geographic contexts, including urban areas, and is particularly relevant to cities and regions using public transportation systems. The project's impact is to enhance mobility and accessibility, primarily in the transportation sector.",
        "label": "negative",
    },
    {
        "name": "gitlab-org/gitlab-runner",
        "text": "No domain-level information available for SDG classification. Project name: GitLab Runner.",
        "label": "negative",
    },
    {
        "name": "firecrawl/firecrawl",
        "text": "Firecrawl addresses the sector of data extraction and web scraping, enabling users to access and process web content at scale. The project's beneficiaries include AI agents, researchers, and developers seeking to gather data from the web. Firecrawl operates in the context of the digital economy, facilitating the extraction of web data for various applications. The project's impact is to provide a scalable and reliable solution for web data extraction, which can be used to support research, development, and innovation. Firecrawl's features and integrations cater to the needs of various stakeholders, including AI agents, developers, and researchers. The project's open-source nature and availability as a cloud service make it accessible to a wide range of users.",
        "label": "negative",
    },
    {
        "name": "citylearn-project/CityLearn",
        "text": "CityLearn is an open source Farama Foundation Gymnasium environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response .CityLearn addresses energy management in urban areas, focusing on building energy coordination and demand response. The project benefits communities and cities by facilitating standardized evaluation of algorithms for efficient energy use. It operates in the context of urban energy systems, aiming to reduce energy consumption and greenhouse gas emissions. CityLearn impacts the energy sector, addressing SDG 7 (Affordable and Clean Energy) and indirectly contributing to SDG 11 (Sustainable Cities and Communities). The project's energy models and simulation tools support the development of sustainable urban energy systems.",
        "label": "positive",
    },
    {
        "name": "trapper-project/trapper",
        "text": "Camera trapping projects in ecological research face significant data management challenges due to the large and complex multimedia datasets generated. TRAPPER addresses this issue by providing an open-source web application that uses spatially enabled data and handles various media types. The platform supports collaborative work and facilitates data sharing among users, promoting the reuse of data and efficient querying of specific subsets. TRAPPER operates in the context of ecological research and conservation, primarily benefiting scientists and ecologists working in wildlife management and conservation. The project's impact is to improve data management and accessibility in camera trapping studies, ultimately contributing to more effective conservation efforts. TRAPPER addresses the sectors of environmental conservation and research, with a focus on wildlife management and data reuse.",
        "label": "positive",
    },
    {
        "name": "OpenMRS/openmrs-core",
        "text": "Healthcare delivery in resource-constrained environments is improved by OpenMRS, a patient-based medical record system. The system provides a free, customizable electronic medical record system for providers. OpenMRS aims to enhance healthcare delivery in low-resource settings by coordinating a global community that creates a robust, scalable, user-driven, open-source medical record system platform. The project addresses the health sector, focusing on improving healthcare outcomes in underserved communities. OpenMRS is used in various geographic contexts, including low-income countries and resource-constrained environments. The project's impact is to increase access to quality healthcare services, particularly in areas with limited resources.",
        "label": "positive",
    },
    # ── Add ~10 more from your 141-project eval set here, using cached text ──
]


def query_ge_lab(text: str) -> dict:
    response = requests.post(GE_LAB_URL, json={"text": text}, timeout=60)
    response.raise_for_status()
    payload = response.json()

    scores_obj = payload.get("scores")
    if scores_obj is None and isinstance(payload.get("data"), dict):
        scores_obj = payload["data"].get("scores")

    if not isinstance(scores_obj, dict):
        raise TypeError(f"Unexpected response shape: {payload}")

    return scores_obj


def main():
    for proj in LABELED_PROJECTS:
        text = proj["text"].strip()
        if not text:
            print(f"{proj['name']:35s} SKIPPED (empty text)")
            continue

        try:
            scores = query_ge_lab(text)
        except Exception as exc:
            print(f"{proj['name']:35s} ERROR: {exc}")
            continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        top3 = ", ".join(f"{name}={score:.3f}" for name, score in ranked[:3])

        print(f"{proj['name']:35s} label={proj['label']:8s} top={top_label} ({top_score:.3f})  |  top3: {top3}")


if __name__ == "__main__":
    main()
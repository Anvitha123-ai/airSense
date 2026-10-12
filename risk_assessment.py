"""
AirSense – Personal Air Pollution Risk Assessment Module
Transparent, rule-based environmental health risk calculation system.
Designed for educational & decision-support purposes (non-clinical).
"""

# Risk Categories & Guidance
RISK_CATEGORIES = [
    {
        "min": 0, "max": 20,
        "level": "Low",
        "color": "#10b981", # green
        "badge_bg": "#d1fae5", "badge_text": "#065f46",
        "general_guidance": "Air quality impact on personal sensitivity is minimal. Normal outdoor activity can generally continue.",
        "precautions": [
            "Enjoy outdoor recreation and exercise freely.",
            "Keep monitoring AQI if planning extensive outdoor activities.",
            "Maintain general wellness habits."
        ]
    },
    {
        "min": 21, "max": 40,
        "level": "Mild",
        "color": "#3b82f6", # blue
        "badge_bg": "#dbeafe", "badge_text": "#1e40af",
        "general_guidance": "Slight risk for highly sensitive individuals during extended exposure. General population remains unaffected.",
        "precautions": [
            "Sensitive individuals should take occasional breaks during heavy outdoor exertion.",
            "Consider tracking hourly AQI changes if commuting in peak traffic.",
            "Stay well hydrated during outdoor tasks."
        ]
    },
    {
        "min": 41, "max": 60,
        "level": "Moderate",
        "color": "#f59e0b", # amber
        "badge_bg": "#fef3c7", "badge_text": "#92400e",
        "general_guidance": "Noticeable risk of discomfort for vulnerable groups. Consider reducing prolonged outdoor exposure when pollution is high.",
        "precautions": [
            "Individuals with asthma or heart conditions should carry prescribed rescue inhalers/medications.",
            "Shift strenuous athletic training indoors or to earlier morning hours.",
            "Close windows facing busy traffic corridors during rush hours."
        ]
    },
    {
        "min": 61, "max": 80,
        "level": "High",
        "color": "#ea580c", # orange/red
        "badge_bg": "#ffedd5", "badge_text": "#9a3412",
        "general_guidance": "Heightened health risk. Consider limiting prolonged or strenuous outdoor activity during high pollution periods.",
        "precautions": [
            "Wear a well-fitted particulate respirator (N95/FFP2) when stepping outdoors.",
            "Avoid strenuous jogging, cycling, or physical labor outdoors.",
            "Use HEPA air purifiers indoors and keep living spaces sealed.",
            "Seniors and children should minimize outdoor playground or transit time."
        ]
    },
    {
        "min": 81, "max": 100,
        "level": "Very High",
        "color": "#dc2626", # deep red
        "badge_bg": "#fee2e2", "badge_text": "#991b1b",
        "general_guidance": "Severe personal exposure risk. Consider avoiding unnecessary prolonged outdoor exposure and strictly following local air-quality guidance.",
        "precautions": [
            "Stay indoors with filtered ventilation as much as possible.",
            "Strictly avoid all outdoor exercise or unnecessary travel.",
            "High-risk individuals experiencing shortness of breath or chest tightness should seek prompt medical consultation.",
            "Follow official public health emergency directives."
        ]
    }
]

HEALTH_CONDITIONS = [
    {
        "id": "asthma",
        "name": "Asthma",
        "weight": 14.0,
        "description": "Hyperreactive airway disease exacerbated by particulate matter and irritant gases."
    },
    {
        "id": "respiratory",
        "name": "Respiratory Sensitivity / COPD",
        "weight": 12.0,
        "description": "Chronic bronchitis, allergic rhinitis, or reduced lung function."
    },
    {
        "id": "cardiovascular",
        "name": "Heart-Related Conditions",
        "weight": 12.0,
        "description": "Hypertension, ischemic heart disease, or past cardiac events."
    },
    {
        "id": "elderly",
        "name": "Older Age (Vulnerability)",
        "weight": 8.0,
        "description": "Age-related reduction in physiological clearance mechanisms."
    },
    {
        "id": "other_sensitive",
        "name": "Other Pollution-Sensitive Conditions",
        "weight": 8.0,
        "description": "Immune deficiency, pregnancy, post-viral respiratory recovery, or seasonal allergies."
    }
]


def calculate_personal_risk(
    age: float,
    aqi: float,
    pm25: float,
    pm10: float,
    exposure_hours: float,
    activity_level: str,
    selected_conditions: list
) -> dict:
    """
    Computes a transparent, weighted Personal Air Pollution Risk Score (0 - 100%).
    
    Weights allocation:
    - Ambient AQI Subscore: 30 pts
    - Particulate Concentration (PM2.5 / PM10) Subscore: 20 pts
    - Age Group Vulnerability Subscore: 15 pts
    - Outdoor Exposure Duration Subscore: 15 pts
    - Physical Activity Level Subscore: 10 pts
    - Selected Health Conditions Subscore: 10 pts (capped at 10 pts)
    Total base sum = 100 pts.
    """
    contributions = {}
    increasing_factors = []
    reducing_factors = []
    
    # 1. Ambient AQI Score (0 - 30 pts)
    # 0 -> 0 pts, 100 -> 9 pts, 200 -> 18 pts, 300 -> 24 pts, 400+ -> 30 pts
    aqi_clamped = max(0.0, min(500.0, float(aqi)))
    if aqi_clamped <= 50:
        aqi_score = (aqi_clamped / 50.0) * 4.0
        reducing_factors.append(f"Low ambient AQI ({int(aqi_clamped)}) reduces atmospheric exposure.")
    elif aqi_clamped <= 100:
        aqi_score = 4.0 + ((aqi_clamped - 50.0) / 50.0) * 5.0
        reducing_factors.append(f"Moderate ambient AQI ({int(aqi_clamped)}) maintains baseline exposure.")
    elif aqi_clamped <= 200:
        aqi_score = 9.0 + ((aqi_clamped - 100.0) / 100.0) * 9.0
        increasing_factors.append(f"Elevated AQI ({int(aqi_clamped)}) enters Moderate pollution band.")
    elif aqi_clamped <= 300:
        aqi_score = 18.0 + ((aqi_clamped - 200.0) / 100.0) * 6.0
        increasing_factors.append(f"High AQI ({int(aqi_clamped)}) in Poor/Unhealthy range.")
    else:
        aqi_score = 24.0 + min(6.0, ((aqi_clamped - 300.0) / 200.0) * 6.0)
        increasing_factors.append(f"Severe AQI ({int(aqi_clamped)}) indicates hazardous air.")
    contributions["Ambient AQI"] = round(aqi_score, 1)

    # 2. PM2.5 & PM10 Concentration Score (0 - 20 pts)
    # PM2.5 CPCB 24h standard = 60 ug/m3, PM10 standard = 100 ug/m3
    pm25_ratio = max(0.0, float(pm25)) / 60.0
    pm10_ratio = max(0.0, float(pm10)) / 100.0
    combined_pm_ratio = (pm25_ratio * 0.7) + (pm10_ratio * 0.3)
    
    if combined_pm_ratio <= 0.5:
        pm_score = combined_pm_ratio * 6.0
        reducing_factors.append(f"Clean particulate levels (PM2.5: {pm25:.1f} ug/m3, PM10: {pm10:.1f} ug/m3).")
    elif combined_pm_ratio <= 1.0:
        pm_score = 3.0 + (combined_pm_ratio - 0.5) * 8.0
    elif combined_pm_ratio <= 2.5:
        pm_score = 7.0 + ((combined_pm_ratio - 1.0) / 1.5) * 7.0
        increasing_factors.append(f"Particulate concentrations exceed safe CPCB standards (PM2.5: {pm25:.1f} ug/m3).")
    else:
        pm_score = 14.0 + min(6.0, ((combined_pm_ratio - 2.5) / 2.5) * 6.0)
        increasing_factors.append(f"Extreme fine particulate concentration (PM2.5: {pm25:.1f} ug/m3).")
    contributions["Particulate Matter (PM2.5/PM10)"] = round(pm_score, 1)

    # 3. Age Group Sensitivity (0 - 15 pts)
    # High vulnerability in young children (<12) and seniors (>60)
    age_val = max(0.0, min(120.0, float(age)))
    if age_val < 5:
        age_score = 14.0
        increasing_factors.append("Early childhood (<5 years) with developing respiratory system.")
    elif age_val <= 12:
        age_score = 10.0
        increasing_factors.append("Childhood age bracket with higher ventilation-to-body mass ratio.")
    elif age_val <= 18:
        age_score = 5.0
    elif age_val <= 50:
        age_score = 2.0
        reducing_factors.append("Adult demographic (19-50) with mature respiratory resilience.")
    elif age_val <= 65:
        age_score = 8.0
        increasing_factors.append("Older adult age bracket (51-65).")
    else:
        age_score = 14.0
        increasing_factors.append(f"Senior age ({int(age_val)} years) with heightened vulnerability to pollution stress.")
    contributions["Age Sensitivity"] = round(age_score, 1)

    # 4. Exposure Duration (0 - 15 pts)
    hours = max(0.0, min(24.0, float(exposure_hours)))
    if hours <= 0.5:
        exp_score = 2.0
        reducing_factors.append("Minimal outdoor exposure (under 30 mins) limits cumulative pollutant intake.")
    elif hours <= 1.5:
        exp_score = 5.0
        reducing_factors.append(f"Moderate exposure duration ({hours:.1f} hours).")
    elif hours <= 4.0:
        exp_score = 9.0
        increasing_factors.append(f"Extended outdoor exposure ({hours:.1f} hours).")
    elif hours <= 8.0:
        exp_score = 12.5
        increasing_factors.append(f"Prolonged exposure ({hours:.1f} hours) increases inhalation burden.")
    else:
        exp_score = 15.0
        increasing_factors.append(f"Continuous outdoor exposure ({hours:.1f} hours).")
    contributions["Exposure Duration"] = round(exp_score, 1)

    # 5. Activity Level (0 - 10 pts)
    # Higher exertion multiplies ventilation rate by 3x - 5x
    act_lower = str(activity_level).strip().lower()
    if "low" in act_lower or "sedentary" in act_lower or "resting" in act_lower:
        act_score = 2.0
        reducing_factors.append("Low physical exertion (walking/resting) keeps minute ventilation low.")
    elif "moderate" in act_lower or "normal" in act_lower:
        act_score = 6.0
        increasing_factors.append("Moderate activity increases respiratory rate moderately.")
    else:  # High / Vigorous
        act_score = 10.0
        increasing_factors.append("Vigorous activity (running/cycling/sports) significantly multiplies deep lung deposition.")
    contributions["Activity Intensity"] = round(act_score, 1)

    # 6. Selected Health Conditions (0 - 10 pts max capped, individual weights scale up)
    selected_clean = [str(c).strip().lower() for c in selected_conditions if c]
    has_none = any("none" in c for c in selected_clean) or len(selected_clean) == 0
    
    cond_score = 0.0
    matched_conditions = []
    
    if not has_none:
        for cond in HEALTH_CONDITIONS:
            if cond["id"] in selected_clean or any(cond["name"].lower() in c for c in selected_clean):
                cond_score += cond["weight"]
                matched_conditions.append(cond["name"])
        
        # Scale to max 10 pts
        cond_score = min(10.0, cond_score)
        if matched_conditions:
            increasing_factors.append(f"Identified pre-existing factors: {', '.join(matched_conditions)}.")
    else:
        cond_score = 0.0
        reducing_factors.append("No pre-existing respiratory or cardiovascular conditions reported.")
        
    contributions["Health Factors"] = round(cond_score, 1)

    # Sum total score (0 - 100%)
    total_score = min(100.0, max(0.0, sum(contributions.values())))
    total_score_int = int(round(total_score))

    # Match Category
    matched_cat = RISK_CATEGORIES[0]
    for cat in RISK_CATEGORIES:
        if cat["min"] <= total_score_int <= cat["max"]:
            matched_cat = cat
            break

    # Top contributing factors ranked
    ranked_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    return {
        "risk_score": total_score_int,
        "risk_level": matched_cat["level"],
        "color": matched_cat["color"],
        "badge_bg": matched_cat["badge_bg"],
        "badge_text": matched_cat["badge_text"],
        "general_guidance": matched_cat["general_guidance"],
        "precautions": matched_cat["precautions"],
        "contributions": contributions,
        "ranked_contributions": ranked_contributions,
        "increasing_factors": increasing_factors,
        "reducing_factors": reducing_factors,
        "disclaimer": (
            "NOTICE: The Estimated Personal Air Pollution Risk Score is an educational and analytical "
            "decision-support estimate produced by the AirSense data science system. It does NOT "
            "constitute a medical diagnosis, clinical prognosis, or personalized healthcare advice. "
            "Actual biological health effects vary between individuals. Always consult qualified "
            "physicians and follow regional public health advisories for health-related concerns."
        )
    }

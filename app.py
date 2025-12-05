#!/usr/bin/env python3
"""
Biosurveillance System Comparison Tool

A Streamlit app for comparing different biosurveillance configurations
and exploring what it takes to achieve reliable pandemic early warning.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# Core Simulation Classes (from simulator.py)
# =============================================================================

GLOBAL_POPULATION = 8.2e9
MAX_CUMULATIVE_INCIDENCE = 0.3


class DetectionMode(Enum):
    MONITORING = "monitoring"
    DISCOVERY = "discovery"


# Swab RA distribution from NAO
SWAB_RA_DISTRIBUTION = [
    5e-6, 5e-6, 6e-6, 7e-6, 1e-5, 1e-5, 2e-5, 3e-5, 3e-5, 3e-5, 4e-5,
    3e-4, 3e-4, 3e-4, 3e-4, 5e-4, 6e-4, 1e-3, 4e-3, 9e-3, 1e-2, 1e-2,
    2e-2, 3e-2, 4e-2, 5e-2, 5e-2, 6e-2, 2e-1, 2e-1, 3e-1, 3e-1, 4e-1,
    6e-1, 6e-1, 7e-1, 2e-7, 9e-7, 2e-5, 1e-5, 1e-5, 7e-5, 5e-5, 1e-2,
    6e-6, 2e-5, 9e-5, 6e-4, 3e-4, 4e-6, 2e-3, 3e-2, 6e-5, 3e-4, 8e-2,
    2e-4, 2e-4, 2e-4, 1e-6, 3e-5, 2e-4, 1e-5, 3e-5, 1e-3,
]
AIRPLANE_RA = 1.4e-6


@dataclass
class PathogenParams:
    doubling_time_days: float = 3.0
    cv_doubling_time: float = 0.1
    shedding_duration_days: float = 5.0
    sigma_shedding_duration: float = 0.05
    genome_length_bp: int = 13_000
    insert_length_bp: int = 170


@dataclass 
class SamplingStrategy:
    name: str
    catchment_size: int
    processing_delay_days: float
    daily_read_depth: float
    num_sites: int = 1
    ra_mode: str = "prevalence"
    ra_at_1pct_prevalence: float = 1e-7
    ra_per_sick_distribution: Optional[List[float]] = None
    sigma_ra: float = 0.5


@dataclass
class SystemConfig:
    name: str
    strategies: List[SamplingStrategy]
    min_samples_with_reads: int = 2
    min_total_reads: int = 2
    target_coverage: float = 2.0
    
    def get_reads_for_discovery(self, genome_length: int, insert_length: int) -> float:
        return self.target_coverage * genome_length / insert_length


class BiosurveillanceSimulator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def simulate_one(
        self,
        system: SystemConfig,
        pathogen: PathogenParams,
        mode: DetectionMode = DetectionMode.MONITORING,
    ) -> Dict:
        doubling_time = max(0.5, self.rng.normal(
            pathogen.doubling_time_days,
            pathogen.cv_doubling_time * pathogen.doubling_time_days
        ))
        
        shedding_duration = self.rng.lognormal(
            np.log(pathogen.shedding_duration_days),
            pathogen.sigma_shedding_duration
        )
        
        r = np.log(2) / doubling_time
        growth_factor = np.exp(r)
        cumulative_incidence = 1.0 / GLOBAL_POPULATION
        
        if mode == DetectionMode.DISCOVERY:
            reads_needed = system.get_reads_for_discovery(
                pathogen.genome_length_bp, pathogen.insert_length_bp
            )
        else:
            reads_needed = system.min_total_reads
        
        total_samples_with_reads = 0
        total_reads = 0
        
        delay_factors = [
            growth_factor ** s.processing_delay_days 
            for s in system.strategies
        ]
        
        day = 0
        while True:
            day += 1
            cumulative_incidence *= growth_factor
            daily_incidence = cumulative_incidence * (1 - 1/growth_factor)
            prob_shedding = self._prob_currently_shedding(
                daily_incidence, shedding_duration, growth_factor
            )
            
            for i, strategy in enumerate(system.strategies):
                if strategy.catchment_size == 0 or strategy.num_sites == 0:
                    continue
                
                # Process each site independently
                for site_idx in range(strategy.num_sites):
                    n_shedding = self.rng.binomial(
                        strategy.catchment_size,
                        min(1.0, prob_shedding)
                    )
                    
                    if n_shedding > 0:
                        if strategy.ra_mode == "per_person" and strategy.ra_per_sick_distribution:
                            if n_shedding > len(strategy.ra_per_sick_distribution) * 3:
                                avg_ra = np.mean(strategy.ra_per_sick_distribution)
                            else:
                                sampled_ras = self.rng.choice(
                                    strategy.ra_per_sick_distribution, 
                                    size=n_shedding,
                                    replace=True
                                )
                                avg_ra = np.mean(sampled_ras)
                            
                            sample_prevalence = n_shedding / strategy.catchment_size
                            relative_abundance = sample_prevalence * avg_ra
                        else:
                            sample_prevalence = n_shedding / strategy.catchment_size
                            relative_abundance = strategy.ra_at_1pct_prevalence * (sample_prevalence / 0.01)
                        
                        if strategy.sigma_ra > 0:
                            noise = self.rng.lognormal(0, strategy.sigma_ra)
                            relative_abundance = relative_abundance * noise
                        
                        expected_reads = strategy.daily_read_depth * relative_abundance
                        actual_reads = self.rng.poisson(expected_reads) if expected_reads < 1e9 else int(expected_reads)
                        
                        if actual_reads > 0:
                            total_samples_with_reads += 1
                            total_reads += actual_reads
                
                if (total_samples_with_reads >= system.min_samples_with_reads and
                    total_reads >= reads_needed):
                    avg_delay_factor = np.mean(delay_factors)
                    detected_incidence = cumulative_incidence * avg_delay_factor
                    return {
                        'cum_incidence': detected_incidence,
                        'day': day,
                        'doubling_time': doubling_time,
                        'detected': True,
                    }
            
            if cumulative_incidence > MAX_CUMULATIVE_INCIDENCE or day > 365 * 5:
                return {
                    'cum_incidence': 1.0,
                    'day': day,
                    'doubling_time': doubling_time,
                    'detected': False,
                }
    
    def _prob_currently_shedding(self, daily_incidence, shedding_duration, growth_factor):
        prob = 0.0
        effective = daily_incidence
        for _ in range(int(shedding_duration)):
            prob += effective
            effective /= growth_factor
        return min(1.0, prob)
    
    def run_simulations(self, system, pathogen, n_simulations=1000, mode=DetectionMode.MONITORING, progress_callback=None):
        results = []
        for i in range(n_simulations):
            if progress_callback and i % max(1, n_simulations // 100) == 0:
                progress_callback(i / n_simulations)
            results.append(self.simulate_one(system, pathogen, mode))
        if progress_callback:
            progress_callback(1.0)
        return results
    
    def analyze_results(self, results, p_bad=0.3, t_gov=45.0, pathogen=None):
        incidences = np.array([r['cum_incidence'] for r in results])
        detected = np.array([r['detected'] for r in results])
        doubling_times = np.array([r['doubling_time'] for r in results])
        
        percentiles = [25, 50, 75, 90, 95, 99]
        
        analysis = {
            'percentiles': {p: float(np.percentile(incidences, p)) for p in percentiles},
            'percentiles_per_100k': {p: float(np.percentile(incidences, p)) * 100_000 for p in percentiles},
            'mean_incidence': float(np.mean(incidences)),
            'detection_rate': float(np.mean(detected)),
            'fraction_before_p_bad': float(np.mean(incidences < p_bad)),
        }
        
        valid = incidences < p_bad
        if np.any(valid):
            inc_valid = incidences[valid]
            dt_valid = doubling_times[valid]
            doublings_to_pbad = np.log(p_bad / inc_valid) / np.log(2)
            time_to_pbad = doublings_to_pbad * dt_valid
            
            analysis['time_slack'] = {
                'mean_days': float(np.mean(time_to_pbad)),
                'percentiles': {p: float(np.percentile(time_to_pbad, p)) for p in percentiles},
            }
            analysis['fraction_with_t_gov'] = float(np.mean(time_to_pbad >= t_gov))
        else:
            analysis['time_slack'] = None
            analysis['fraction_with_t_gov'] = 0.0
        
        return analysis


# =============================================================================
# System Configuration Builders
# =============================================================================

def build_system(name: str, 
                 nwss_catchment: int, nwss_sites: int,
                 swab_catchment: int, swab_sites: int,
                 trit_catchment: int, trit_sites: int,
                 plane_catchment: int, plane_sites: int,
                 nwss_delay: float, swab_delay: float, 
                 trit_delay: float, plane_delay: float) -> SystemConfig:
    """Build a system configuration from UI parameters."""
    return SystemConfig(
        name=name,
        strategies=[
            SamplingStrategy(
                name="NWSS Wastewater",
                catchment_size=nwss_catchment,
                num_sites=nwss_sites,
                processing_delay_days=nwss_delay,
                daily_read_depth=24e9 / max(1, nwss_sites),  # Split reads across sites
                ra_mode="prevalence",
                ra_at_1pct_prevalence=1e-7,
                sigma_ra=0.5,
            ),
            SamplingStrategy(
                name="Nasal Swabs",
                catchment_size=swab_catchment,
                num_sites=swab_sites,
                processing_delay_days=swab_delay,
                daily_read_depth=2e9 / max(1, swab_sites),
                ra_mode="per_person",
                ra_per_sick_distribution=SWAB_RA_DISTRIBUTION,
                sigma_ra=0.05,
            ),
            SamplingStrategy(
                name="Triturators",
                catchment_size=trit_catchment,
                num_sites=trit_sites,
                processing_delay_days=trit_delay,
                daily_read_depth=188e9 / max(1, trit_sites),
                ra_mode="per_person",
                ra_per_sick_distribution=[AIRPLANE_RA],
                sigma_ra=0.5,
            ),
            SamplingStrategy(
                name="Individual Planes",
                catchment_size=plane_catchment,
                num_sites=plane_sites,
                processing_delay_days=plane_delay,
                daily_read_depth=12e9 / max(1, plane_sites),
                ra_mode="per_person",
                ra_per_sick_distribution=[AIRPLANE_RA],
                sigma_ra=0.5,
            ),
        ],
    )


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(
    page_title="Biosurveillance System Comparison",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 Pandemic Early Warning System Comparison")

st.markdown("""
This tool simulates detection of emerging pathogens using different biosurveillance 
strategies. Based on the [Nucleic Acid Observatory's methodology](https://naobservatory.org/).

**Two detection modes:**
- **Monitoring**: Detecting *known* pathogens (need ~2 matching reads)
- **Discovery**: Characterizing *novel* pathogens (need ~2× genome coverage ≈ 153 reads)

**Key insight**: Different strategies scale differently:
- **Wastewater**: Catchment size doesn't help much (prevalence cancels out). More *sites* helps with geographic coverage.
- **Swabs**: Larger catchment helps a lot (need to catch sick people). Signal is strong per sick person.
- **Airplane waste**: Between the two - moderate RA, benefits from more passengers.
""")

# =============================================================================
# Sidebar: Global Parameters
# =============================================================================

st.sidebar.header("🎛️ Simulation Parameters")

n_simulations = st.sidebar.slider(
    "Number of simulations",
    min_value=100, max_value=5000, value=1000, step=100,
    help="More simulations = more accurate results but slower"
)

st.sidebar.subheader("Pathogen Parameters")
doubling_time = st.sidebar.slider(
    "Doubling time (days)",
    min_value=1.0, max_value=14.0, value=3.0, step=0.5,
    help="How fast the pathogen spreads. COVID-19 ≈ 3 days, Influenza ≈ 2-3 days"
)

st.sidebar.subheader("Policy Parameters")
p_bad = st.sidebar.slider(
    "p_bad: 'Game over' threshold",
    min_value=0.05, max_value=0.50, value=0.30, step=0.05,
    format="%.0f%%",
    help="Cumulative incidence at which it's 'too late' to respond effectively"
)

t_gov = st.sidebar.slider(
    "T_gov: Government response time (days)",
    min_value=14.0, max_value=120.0, value=45.0, step=1.0,
    help="How many days lead time does the government need to mount an effective response?"
)

detection_mode = st.sidebar.radio(
    "Detection mode",
    options=["monitoring", "discovery"],
    format_func=lambda x: "Monitoring (known pathogens)" if x == "monitoring" else "Discovery (novel pathogens)",
    help="Monitoring needs ~2 reads, Discovery needs ~153 reads for 2× coverage"
)

seed = st.sidebar.number_input("Random seed", value=42, step=1)

# =============================================================================
# Main Content: System Configuration
# =============================================================================

st.header("📊 System Configuration")

# Default values for current and FY2026 systems
CURRENT_DEFAULTS = {
    "nwss": 200_000, "nwss_sites": 5,
    "swab": 191, "swab_sites": 1,
    "trit": 0, "trit_sites": 0,
    "plane": 0, "plane_sites": 0,
    "nwss_delay": 7.0, "swab_delay": 5.0, "trit_delay": 5.0, "plane_delay": 5.0
}

FY2026_DEFAULTS = {
    "nwss": 500_000, "nwss_sites": 5,
    "swab": 400, "swab_sites": 13,
    "trit": 7_500, "trit_sites": 13,
    "plane": 2_250, "plane_sites": 2,
    "nwss_delay": 2.69, "swab_delay": 2.19, "trit_delay": 2.65, "plane_delay": 2.40
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("System A")
    name_a = st.text_input("Name", "Current System (2024)", key="name_a")
    
    st.markdown("**Catchment per Site × Number of Sites:**")
    c1, c2 = st.columns(2)
    with c1:
        nwss_a = st.number_input("NWSS per site", 0, 5_000_000, CURRENT_DEFAULTS["nwss"], 50_000, key="nwss_a")
        swab_a = st.number_input("Swabs per site", 0, 10_000, CURRENT_DEFAULTS["swab"], 50, key="swab_a")
        trit_a = st.number_input("Trit. per site", 0, 50_000, CURRENT_DEFAULTS["trit"], 1_000, key="trit_a")
        plane_a = st.number_input("Planes per site", 0, 20_000, CURRENT_DEFAULTS["plane"], 500, key="plane_a")
    with c2:
        nwss_sites_a = st.number_input("NWSS sites", 0, 100, CURRENT_DEFAULTS["nwss_sites"], 1, key="nwss_sites_a")
        swab_sites_a = st.number_input("Swab sites", 0, 50, CURRENT_DEFAULTS["swab_sites"], 1, key="swab_sites_a")
        trit_sites_a = st.number_input("Trit. sites", 0, 50, CURRENT_DEFAULTS["trit_sites"], 1, key="trit_sites_a")
        plane_sites_a = st.number_input("Plane sites", 0, 20, CURRENT_DEFAULTS["plane_sites"], 1, key="plane_sites_a")
    
    st.markdown("**Processing Delays (days):**")
    delay_nwss_a = st.number_input("NWSS delay", 0.5, 14.0, CURRENT_DEFAULTS["nwss_delay"], 0.5, key="dnwss_a")
    delay_swab_a = st.number_input("Swab delay", 0.5, 14.0, CURRENT_DEFAULTS["swab_delay"], 0.5, key="dswab_a")
    delay_trit_a = st.number_input("Trit. delay", 0.5, 14.0, CURRENT_DEFAULTS["trit_delay"], 0.5, key="dtrit_a")
    delay_plane_a = st.number_input("Plane delay", 0.5, 14.0, CURRENT_DEFAULTS["plane_delay"], 0.5, key="dplane_a")
    
    total_a = nwss_a * nwss_sites_a + swab_a * swab_sites_a + trit_a * trit_sites_a + plane_a * plane_sites_a
    st.info(f"**Total catchment**: {total_a:,}")

with col2:
    st.subheader("System B")
    name_b = st.text_input("Name", "FY2026 Biothreat Radar", key="name_b")
    
    st.markdown("**Catchment per Site × Number of Sites:**")
    c1, c2 = st.columns(2)
    with c1:
        nwss_b = st.number_input("NWSS per site", 0, 5_000_000, FY2026_DEFAULTS["nwss"], 50_000, key="nwss_b")
        swab_b = st.number_input("Swabs per site", 0, 10_000, FY2026_DEFAULTS["swab"], 50, key="swab_b")
        trit_b = st.number_input("Trit. per site", 0, 50_000, FY2026_DEFAULTS["trit"], 1_000, key="trit_b")
        plane_b = st.number_input("Planes per site", 0, 20_000, FY2026_DEFAULTS["plane"], 500, key="plane_b")
    with c2:
        nwss_sites_b = st.number_input("NWSS sites", 0, 100, FY2026_DEFAULTS["nwss_sites"], 1, key="nwss_sites_b")
        swab_sites_b = st.number_input("Swab sites", 0, 50, FY2026_DEFAULTS["swab_sites"], 1, key="swab_sites_b")
        trit_sites_b = st.number_input("Trit. sites", 0, 50, FY2026_DEFAULTS["trit_sites"], 1, key="trit_sites_b")
        plane_sites_b = st.number_input("Plane sites", 0, 20, FY2026_DEFAULTS["plane_sites"], 1, key="plane_sites_b")
    
    st.markdown("**Processing Delays (days):**")
    delay_nwss_b = st.number_input("NWSS delay", 0.5, 14.0, FY2026_DEFAULTS["nwss_delay"], 0.5, key="dnwss_b")
    delay_swab_b = st.number_input("Swab delay", 0.5, 14.0, FY2026_DEFAULTS["swab_delay"], 0.5, key="dswab_b")
    delay_trit_b = st.number_input("Trit. delay", 0.5, 14.0, FY2026_DEFAULTS["trit_delay"], 0.5, key="dtrit_b")
    delay_plane_b = st.number_input("Plane delay", 0.5, 14.0, FY2026_DEFAULTS["plane_delay"], 0.5, key="dplane_b")
    
    total_b = nwss_b * nwss_sites_b + swab_b * swab_sites_b + trit_b * trit_sites_b + plane_b * plane_sites_b
    st.info(f"**Total catchment**: {total_b:,}")

# =============================================================================
# Run Simulations
# =============================================================================

if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
    
    # Build systems
    system_a = build_system(
        name_a, 
        nwss_a, nwss_sites_a, swab_a, swab_sites_a,
        trit_a, trit_sites_a, plane_a, plane_sites_a,
        delay_nwss_a, delay_swab_a, delay_trit_a, delay_plane_a
    )
    system_b = build_system(
        name_b,
        nwss_b, nwss_sites_b, swab_b, swab_sites_b,
        trit_b, trit_sites_b, plane_b, plane_sites_b,
        delay_nwss_b, delay_swab_b, delay_trit_b, delay_plane_b
    )
    
    pathogen = PathogenParams(doubling_time_days=doubling_time)
    mode = DetectionMode.MONITORING if detection_mode == "monitoring" else DetectionMode.DISCOVERY
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Run simulations
    simulator = BiosurveillanceSimulator(seed=seed)
    
    status.text(f"Running {name_a}...")
    results_a = simulator.run_simulations(
        system_a, pathogen, n_simulations, mode,
        progress_callback=lambda p: progress_bar.progress(p * 0.5)
    )
    analysis_a = simulator.analyze_results(results_a, p_bad, t_gov, pathogen)
    
    status.text(f"Running {name_b}...")
    results_b = simulator.run_simulations(
        system_b, pathogen, n_simulations, mode,
        progress_callback=lambda p: progress_bar.progress(0.5 + p * 0.5)
    )
    analysis_b = simulator.analyze_results(results_b, p_bad, t_gov, pathogen)
    
    progress_bar.progress(1.0)
    status.text("Complete!")
    
    # =============================================================================
    # Results Display
    # =============================================================================
    
    st.header("📈 Results")
    
    # Key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(name_a)
        median_a = analysis_a['percentiles_per_100k'][50]
        t_gov_pct_a = analysis_a['fraction_with_t_gov'] * 100
        
        st.metric("Median detection", f"{median_a:.2f} per 100k")
        st.metric(f"Has {t_gov:.0f} day lead time", f"{t_gov_pct_a:.1f}%")
        if analysis_a['time_slack']:
            st.metric("Mean lead time", f"{analysis_a['time_slack']['mean_days']:.1f} days")
    
    with col2:
        st.subheader(name_b)
        median_b = analysis_b['percentiles_per_100k'][50]
        t_gov_pct_b = analysis_b['fraction_with_t_gov'] * 100
        
        st.metric("Median detection", f"{median_b:.2f} per 100k")
        st.metric(f"Has {t_gov:.0f} day lead time", f"{t_gov_pct_b:.1f}%")
        if analysis_b['time_slack']:
            st.metric("Mean lead time", f"{analysis_b['time_slack']['mean_days']:.1f} days")
    
    # Comparison summary
    if median_a > 0 and median_b > 0:
        ratio = median_a / median_b
        if ratio > 1:
            st.success(f"**{name_b}** detects **{ratio:.1f}× earlier** than {name_a}")
        else:
            st.success(f"**{name_a}** detects **{1/ratio:.1f}× earlier** than {name_b}")
    
    # Detailed percentiles table
    st.subheader("Detection Thresholds (per 100k)")
    
    percentile_data = []
    for p in [25, 50, 75, 90, 95, 99]:
        percentile_data.append({
            "Percentile": f"{p}th",
            name_a: f"{analysis_a['percentiles_per_100k'][p]:.2f}",
            name_b: f"{analysis_b['percentiles_per_100k'][p]:.2f}",
        })
    
    st.table(pd.DataFrame(percentile_data))
    
    # Distribution plot
    st.subheader("Detection Distribution")
    
    inc_a = [r['cum_incidence'] * 100_000 for r in results_a if r['detected']]
    inc_b = [r['cum_incidence'] * 100_000 for r in results_b if r['detected']]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=inc_a, name=name_a, opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=inc_b, name=name_b, opacity=0.7, nbinsx=50))
    fig.update_layout(
        barmode='overlay',
        xaxis_title="Cumulative incidence at detection (per 100k)",
        yaxis_title="Count",
        title="Distribution of Detection Thresholds"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Time slack distribution
    if analysis_a['time_slack'] and analysis_b['time_slack']:
        st.subheader("Lead Time Distribution")
        
        times_a = []
        times_b = []
        for r in results_a:
            if r['detected'] and r['cum_incidence'] < p_bad:
                doublings = np.log(p_bad / r['cum_incidence']) / np.log(2)
                times_a.append(doublings * r['doubling_time'])
        for r in results_b:
            if r['detected'] and r['cum_incidence'] < p_bad:
                doublings = np.log(p_bad / r['cum_incidence']) / np.log(2)
                times_b.append(doublings * r['doubling_time'])
        
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=times_a, name=name_a, opacity=0.7, nbinsx=50))
        fig2.add_trace(go.Histogram(x=times_b, name=name_b, opacity=0.7, nbinsx=50))
        fig2.add_vline(x=t_gov, line_dash="dash", line_color="red", 
                       annotation_text=f"T_gov = {t_gov:.0f} days")
        fig2.update_layout(
            barmode='overlay',
            xaxis_title=f"Days until {p_bad:.0%} infected",
            yaxis_title="Count",
            title="Lead Time Before 'Game Over'"
        )
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# Exploration Mode: Find Optimal Configuration
# =============================================================================

st.header("🔍 Explore: What Does It Take?")

st.markdown("""
This section explores different catchment size combinations to find configurations 
that achieve a target level of reliability (e.g., 80% chance of detecting with 
sufficient lead time).
""")

with st.expander("Run Exploration Analysis", expanded=False):

    target_reliability = st.slider(
        "Target reliability",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
        help="Target fraction of runs with sufficient lead time"
    )   

st.write(f"Selected target reliability: {target_reliability:.0%}")
    
    exploration_sims = st.slider(
        "Simulations per configuration",
        min_value=100, max_value=1000, value=300, step=100
    )
    
    if st.button("🔬 Run Exploration"):
        pathogen = PathogenParams(doubling_time_days=doubling_time)
        mode = DetectionMode.MONITORING if detection_mode == "monitoring" else DetectionMode.DISCOVERY
        
        # Test different scaling factors
        scales = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        
        results_grid = []
        progress = st.progress(0)
        status = st.empty()
        
        total_configs = len(scales) ** 2
        config_num = 0
        
        for nwss_scale in scales:
            for swab_scale in scales:
                config_num += 1
                progress.progress(config_num / total_configs)
                status.text(f"Testing NWSS×{nwss_scale}, Swabs×{swab_scale}...")
                
                system = build_system(
                    f"NWSS×{nwss_scale}, Swabs×{swab_scale}",
                    int(FY2026_DEFAULTS["nwss"] * nwss_scale),  # per site
                    FY2026_DEFAULTS["nwss_sites"],
                    int(FY2026_DEFAULTS["swab"] * swab_scale),  # per site
                    FY2026_DEFAULTS["swab_sites"],
                    FY2026_DEFAULTS["trit"],
                    FY2026_DEFAULTS["trit_sites"],
                    FY2026_DEFAULTS["plane"],
                    FY2026_DEFAULTS["plane_sites"],
                    FY2026_DEFAULTS["nwss_delay"],
                    FY2026_DEFAULTS["swab_delay"],
                    FY2026_DEFAULTS["trit_delay"],
                    FY2026_DEFAULTS["plane_delay"],
                )
                
                sim = BiosurveillanceSimulator(seed=seed)
                res = sim.run_simulations(system, pathogen, exploration_sims, mode)
                analysis = sim.analyze_results(res, p_bad, t_gov, pathogen)
                
                results_grid.append({
                    'nwss_scale': nwss_scale,
                    'swab_scale': swab_scale,
                    'nwss_catchment': int(FY2026_DEFAULTS["nwss"] * nwss_scale),
                    'swab_catchment': int(FY2026_DEFAULTS["swab"] * swab_scale),
                    'median_per_100k': analysis['percentiles_per_100k'][50],
                    'fraction_with_t_gov': analysis['fraction_with_t_gov'],
                    'meets_target': analysis['fraction_with_t_gov'] >= target_reliability,
                })
        
        status.text("Complete!")
        
        # Display results as heatmap
        df = pd.DataFrame(results_grid)
        
        pivot = df.pivot(
            index='nwss_scale', 
            columns='swab_scale', 
            values='fraction_with_t_gov'
        )
        
        fig = px.imshow(
            pivot * 100,
            labels=dict(x="Swab catchment scale", y="NWSS catchment scale", color="% with T_gov"),
            x=[f"{s}×" for s in scales],
            y=[f"{s}×" for s in scales],
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title=f"Fraction with {t_gov:.0f} day lead time (target: {target_reliability:.0%})"
        )
        fig.add_contour(
            z=pivot.values * 100,
            x=list(range(len(scales))),
            y=list(range(len(scales))),
            contours=dict(
                start=target_reliability * 100,
                end=target_reliability * 100,
                size=1
            ),
            line=dict(width=3, color='black'),
            showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show configurations that meet target
        meets_target = df[df['meets_target']]
        if len(meets_target) > 0:
            st.success(f"Found {len(meets_target)} configurations meeting {target_reliability:.0%} target:")
            st.dataframe(meets_target[['nwss_catchment', 'swab_catchment', 'median_per_100k', 'fraction_with_t_gov']])
        else:
            st.warning(f"No configurations tested achieve {target_reliability:.0%} reliability. Consider increasing catchment scales.")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
**Notes:**
- Detection at lower cumulative incidence = better (more warning time)
- T_gov lead time measures if we detect early enough for effective response
- Based on [NAO's Biothreat Radar methodology](https://naobservatory.org/blog/biothreat_radar/)
""")

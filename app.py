import streamlit as st
import numpy as np
from typing import Dict, List, Optional

# Hard-coded RAI1PCT data (subset - using Rothman-2697049 which NAO uses for wastewater)
WW_RAI1PCT = {
    "MU-11320": {"1e-7": 2,"1e-8": 137,"1e-9": 2,"1.1e-7": 9,"1.1e-8": 254,"1.1e-9": 1,"1.2e-5": 2,"1.2e-7": 9,"1.2e-8": 225,"1.2e-9": 2,"1.3e-6": 1,"1.3e-7": 3,"1.3e-8": 216,"1.3e-9": 1,"1.3e-10": 1,"1.4e-6": 1,"1.4e-7": 6,"1.4e-8": 206,"1.4e-9": 1,"1.4e-10": 1,"1.5e-7": 3,"1.5e-8": 161,"1.5e-9": 2,"1.6e-5": 1,"1.6e-7": 4,"1.6e-8": 158,"1.6e-9": 5,"1.7e-7": 3,"1.7e-8": 125,"1.7e-9": 4,"1.8e-7": 1,"1.8e-8": 112,"1.8e-9": 2,"1.9e-7": 2,"1.9e-8": 111,"1.9e-9": 1,"2e-7": 2,"2e-8": 107,"2e-9": 4,"2.1e-5": 1,"2.1e-7": 1,"2.1e-8": 75,"2.1e-9": 3,"2.2e-7": 1,"2.2e-8": 79,"2.2e-9": 2,"2.3e-8": 69,"2.3e-9": 6,"2.4e-7": 2,"2.4e-8": 55,"2.4e-9": 6,"2.5e-7": 1,"2.5e-8": 51,"2.5e-9": 4,"2.6e-8": 55,"2.6e-9": 4,"2.7e-8": 37,"2.7e-9": 6,"2.8e-8": 35,"2.8e-9": 6,"2.9e-8": 35,"2.9e-9": 7,"3e-6": 1,"3e-8": 33,"3e-9": 7,"3.1e-7": 1,"3.1e-8": 29,"3.1e-9": 8,"3.2e-7": 1,"3.2e-8": 27,"3.2e-9": 5,"3.3e-8": 25,"3.3e-9": 9,"3.4e-8": 29,"3.4e-9": 3,"3.5e-8": 19,"3.5e-9": 5,"3.6e-7": 1,"3.6e-8": 23,"3.6e-9": 3,"3.7e-8": 24,"3.7e-9": 10,"3.8e-6": 2,"3.8e-8": 9,"3.8e-9": 8,"3.9e-6": 1,"3.9e-8": 16,"3.9e-9": 6,"3.9e-10": 1,"4e-8": 18,"4e-9": 16,"4e-10": 1,"4.1e-8": 13,"4.1e-9": 5,"4.2e-8": 13,"4.2e-9": 7,"4.2e-10": 1,"4.3e-7": 2,"4.3e-8": 11,"4.3e-9": 8,"4.4e-8": 9,"4.4e-9": 5,"4.5e-6": 1,"4.5e-8": 8,"4.5e-9": 9,"4.6e-7": 1,"4.6e-8": 5,"4.6e-9": 13,"4.7e-7": 1,"4.7e-8": 6,"4.7e-9": 5,"4.7e-10": 1,"4.8e-7": 1,"4.8e-8": 4,"4.8e-9": 10,"4.9e-8": 6,"4.9e-9": 6,"5e-8": 7,"5e-9": 15,"5.1e-6": 1,"5.1e-8": 8,"5.1e-9": 9,"5.2e-8": 4,"5.2e-9": 10,"5.2e-10": 2,"5.3e-8": 8,"5.3e-9": 8,"5.4e-8": 8,"5.4e-9": 13,"5.5e-7": 1,"5.5e-8": 6,"5.5e-9": 12,"5.6e-8": 4,"5.6e-9": 12,"5.7e-8": 4,"5.7e-9": 6,"5.8e-8": 4,"5.8e-9": 10,"5.9e-7": 1,"5.9e-8": 8,"5.9e-9": 11,"5.9e-10": 1,"5.9e-11": 1,"6e-8": 3,"6e-9": 9,"6.1e-8": 8,"6.1e-9": 11,"6.2e-8": 4,"6.2e-9": 19,"6.3e-8": 7,"6.3e-9": 17,"6.4e-8": 8,"6.4e-9": 19,"6.5e-8": 3,"6.5e-9": 13,"6.6e-8": 3,"6.6e-9": 6,"6.6e-10": 1,"6.7e-6": 1,"6.7e-7": 1,"6.7e-8": 4,"6.7e-9": 18,"6.8e-8": 3,"6.8e-9": 16,"6.9e-8": 6,"6.9e-9": 19,"7e-8": 2,"7e-9": 23,"7.1e-8": 3,"7.1e-9": 13,"7.2e-8": 4,"7.2e-9": 12,"7.3e-8": 2,"7.3e-9": 18,"7.4e-8": 2,"7.4e-9": 21,"7.5e-8": 2,"7.5e-9": 22,"7.6e-8": 2,"7.6e-9": 21,"7.7e-8": 1,"7.7e-9": 21,"7.8e-7": 1,"7.8e-8": 4,"7.8e-9": 18,"7.8e-10": 1,"7.9e-8": 1,"7.9e-9": 15,"8e-8": 4,"8e-9": 26,"8.1e-8": 1,"8.1e-9": 13,"8.1e-10": 1,"8.2e-8": 3,"8.2e-9": 20,"8.3e-8": 4,"8.3e-9": 17,"8.4e-8": 1,"8.4e-9": 28,"8.5e-8": 1,"8.5e-9": 25,"8.6e-7": 1,"8.6e-9": 21,"8.7e-9": 29,"8.8e-8": 4,"8.8e-9": 28,"8.8e-10": 1,"8.9e-7": 1,"8.9e-8": 1,"8.9e-9": 21,"9e-8": 4,"9e-9": 16,"9.1e-9": 31,"9.2e-8": 2,"9.2e-9": 22,"9.3e-8": 1,"9.3e-9": 24,"9.4e-8": 1,"9.4e-9": 23,"9.4e-10": 1,"9.5e-8": 2,"9.5e-9": 20,"9.6e-8": 1,"9.6e-9": 16,"9.6e-10": 1,"9.6e-11": 1,"9.7e-9": 22,"9.8e-9": 28,"9.9e-8": 2,"9.9e-9": 28},
    "Rothman-2697049": {"1e-6": 1,"1e-7": 108,"1e-8": 3,"1.1e-7": 131,"1.1e-8": 22,"1.1e-9": 1,"1.2e-6": 2,"1.2e-7": 100,"1.2e-8": 17,"1.3e-7": 94,"1.3e-8": 20,"1.4e-6": 1,"1.4e-7": 85,"1.4e-8": 25,"1.5e-7": 74,"1.5e-8": 25,"1.5e-9": 1,"1.6e-7": 34,"1.6e-8": 24,"1.7e-7": 37,"1.7e-8": 25,"1.8e-7": 29,"1.8e-8": 36,"1.8e-9": 1,"1.9e-6": 1,"1.9e-7": 21,"1.9e-8": 34,"2e-7": 20,"2e-8": 31,"2e-9": 1,"2.1e-7": 22,"2.1e-8": 34,"2.2e-7": 18,"2.2e-8": 39,"2.3e-7": 12,"2.3e-8": 33,"2.4e-7": 8,"2.4e-8": 31,"2.4e-9": 1,"2.5e-7": 10,"2.5e-8": 48,"2.6e-7": 9,"2.6e-8": 37,"2.6e-9": 1,"2.7e-7": 10,"2.7e-8": 40,"2.8e-7": 7,"2.8e-8": 39,"2.9e-7": 4,"2.9e-8": 34,"3e-7": 4,"3e-8": 47,"3.1e-7": 6,"3.1e-8": 43,"3.2e-7": 4,"3.2e-8": 41,"3.2e-9": 2,"3.3e-7": 4,"3.3e-8": 43,"3.3e-9": 1,"3.4e-7": 5,"3.4e-8": 56,"3.4e-9": 2,"3.5e-7": 4,"3.5e-8": 42,"3.6e-7": 3,"3.6e-8": 41,"3.6e-9": 1,"3.7e-7": 2,"3.7e-8": 41,"3.7e-9": 1,"3.8e-7": 3,"3.8e-8": 56,"3.8e-9": 1,"3.9e-7": 1,"3.9e-8": 38,"4e-7": 1,"4e-8": 46,"4.1e-7": 1,"4.1e-8": 41,"4.1e-9": 1,"4.2e-8": 41,"4.3e-7": 2,"4.3e-8": 35,"4.3e-9": 2,"4.4e-7": 1,"4.4e-8": 36,"4.4e-9": 2,"4.5e-8": 41,"4.6e-8": 38,"4.6e-9": 2,"4.7e-7": 1,"4.7e-8": 53,"4.8e-8": 45,"4.9e-8": 46,"5e-8": 42,"5e-9": 1,"5.1e-7": 1,"5.1e-8": 50,"5.2e-8": 46,"5.3e-8": 44,"5.3e-9": 1,"5.4e-7": 2,"5.4e-8": 42,"5.5e-8": 50,"5.5e-9": 1,"5.6e-7": 1,"5.6e-8": 33,"5.7e-7": 1,"5.7e-8": 47,"5.7e-9": 1,"5.8e-7": 1,"5.8e-8": 45,"5.8e-9": 2,"5.9e-7": 1,"5.9e-8": 24,"5.9e-9": 2,"6e-8": 38,"6.1e-7": 3,"6.1e-8": 48,"6.1e-9": 1,"6.2e-8": 30,"6.2e-9": 1,"6.3e-8": 33,"6.3e-9": 1,"6.4e-8": 39,"6.4e-9": 1,"6.5e-8": 40,"6.6e-8": 41,"6.7e-8": 27,"6.7e-9": 1,"6.8e-7": 1,"6.8e-8": 44,"6.8e-9": 1,"6.9e-8": 34,"7e-8": 38,"7e-9": 1,"7.1e-8": 34,"7.1e-9": 4,"7.2e-8": 36,"7.2e-9": 1,"7.3e-8": 38,"7.3e-9": 2,"7.4e-8": 25,"7.4e-9": 1,"7.5e-8": 40,"7.5e-9": 2,"7.6e-8": 23,"7.6e-9": 3,"7.7e-8": 29,"7.8e-7": 1,"7.8e-8": 27,"7.9e-8": 29,"7.9e-9": 1,"8e-8": 36,"8e-9": 2,"8.1e-8": 27,"8.1e-9": 2,"8.2e-8": 20,"8.2e-9": 1,"8.3e-7": 1,"8.3e-8": 23,"8.4e-7": 1,"8.4e-8": 26,"8.4e-9": 4,"8.5e-7": 1,"8.5e-8": 18,"8.5e-9": 1,"8.6e-7": 1,"8.6e-8": 27,"8.6e-9": 3,"8.7e-8": 19,"8.7e-9": 2,"8.8e-8": 25,"8.8e-9": 1,"8.9e-7": 2,"8.9e-8": 20,"9e-8": 23,"9e-9": 3,"9.1e-8": 16,"9.1e-9": 1,"9.2e-8": 20,"9.2e-9": 2,"9.3e-8": 20,"9.4e-8": 16,"9.4e-9": 3,"9.5e-8": 20,"9.6e-8": 29,"9.6e-9": 2,"9.7e-8": 13,"9.7e-9": 2,"9.8e-8": 19,"9.8e-9": 2,"9.9e-8": 20,"9.9e-9": 1}
}

POPULATION = 8.2e9
MAX_SUPPORTED_CUMULATIVE_INCIDENCE = 0.3

class BiosurveillanceSimulator:
    def __init__(self, params: Dict, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.params = params

    def inverse_transform_sample(self, cdf: Dict[str, float]) -> float:
        total_weight = sum(cdf.values())
        target = self.rng.random() * total_weight
        cumsum = 0.0
        for k, v in cdf.items():
            cumsum += v
            if cumsum >= target:
                return float(k)
        return float(list(cdf.keys())[-1])

    def individual_probability_sick(self, daily_incidence: float,
                                    detectable_days: float,
                                    growth_factor: float) -> float:
        prob = 0.0
        effective = daily_incidence
        for _ in range(int(detectable_days)):
            prob += effective
            effective /= growth_factor
        return prob

    def simulate_ra_sick(self, sample_sick: int, ra_sicks: List[float]) -> float:
        if sample_sick == 0:
            return 0.0
        elif len(ra_sicks) == 1:
            return ra_sicks[0]
        elif sample_sick > len(ra_sicks) * 3:
            return float(np.mean(ra_sicks))
        else:
            return float(np.mean(self.rng.choice(ra_sicks, size=sample_sick)))

    def get_noisy_value(self, mean: float, cv: float) -> float:
        if cv < 1e-6:
            return mean
        return float(self.rng.normal(mean, cv * mean))

    def get_lognormal_value(self, geom_mean: float, sigma: float) -> float:
        if sigma < 1e-6:
            return geom_mean
        return float(self.rng.lognormal(np.log(geom_mean), sigma))

    def simulate_one(self) -> Dict[str, float]:
        day = 0
        doubling_time = self.get_noisy_value(
            self.params["doubling_time"],
            self.params["cv_doubling_time"],
        )
        r = np.log(2.0) / doubling_time
        growth_factor = np.exp(r)
        cumulative_incidence = 1.0 / POPULATION

        detectable_days = self.get_lognormal_value(
            self.params["shedding_duration"],
            self.params["sigma_shedding_duration"],
        )

        ra_sickss = []
        for shedding_values in self.params["shedding_values"]:
            if isinstance(shedding_values, str):
                rai1pct = self.inverse_transform_sample(WW_RAI1PCT[shedding_values])
                per_sick = rai1pct / 0.01
                ra_sickss.append([per_sick])
            else:
                if self.params["sigma_shedding_values"] > 1e-6:
                    bias = self.rng.lognormal(0.0, self.params["sigma_shedding_values"])
                    ra_sickss.append([v * bias for v in shedding_values if v is not None])
                else:
                    ra_sickss.append([v for v in shedding_values if v is not None])

        sample_observations = 0
        total_read_observations = 0  # Track total reads across all samples

        site_infos = [{"sample_sick": 0, "sample_total": 0} for _ in ra_sickss]

        processing_delay_factors = [
            growth_factor ** processing_delay
            for processing_delay in self.params["processing_delays"]
        ]

        # For discovery mode: calculate reads needed for 2x coverage
        # reads_needed = (2 * genome_length) / insert_length * (1 / fraction_useful)
        # But NAO's model uses: genome observed at 2x average coverage
        # This means: total_useful_reads >= 2 * (genome_length / insert_length)
        genome_length = self.params["genome_length_bp"]
        insert_length = self.params["insert_length_bp"]
        target_coverage = self.params.get("target_coverage", 2.0)
        
        # Number of reads needed to achieve target coverage
        # Each read covers insert_length bp, need to cover genome target_coverage times
        reads_for_coverage = target_coverage * genome_length / insert_length

        while True:
            day += 1
            cumulative_incidence *= growth_factor
            daily_incidence = cumulative_incidence * (1.0 - 1.0 / growth_factor)

            for (
                site,
                sample_population,
                ra_sicks,
                sample_depth,
                processing_delay_factor,
            ) in zip(
                site_infos,
                self.params["sample_populations"],
                ra_sickss,
                self.params["sample_depths"],
                processing_delay_factors,
            ):
                prob_sick = self.individual_probability_sick(
                    daily_incidence=daily_incidence,
                    detectable_days=detectable_days,
                    growth_factor=growth_factor,
                )
                prob_sick = max(0.0, min(1.0, prob_sick))

                site["sample_total"] = int(sample_population)
                site["sample_sick"] = self.rng.binomial(site["sample_total"], prob_sick)

                relative_abundance = 0.0
                if site["sample_total"] > 0 and site["sample_sick"] > 0:
                    prevalence = site["sample_sick"] / site["sample_total"]
                    relative_abundance = prevalence * self.simulate_ra_sick(
                        site["sample_sick"], ra_sicks
                    )

                site["sample_sick"] = 0
                site["sample_total"] = 0

                if relative_abundance > 0.0:
                    prob_useful = (
                        self.params["fraction_useful_reads"] * relative_abundance
                    )
                    this_sample_obs = self.rng.poisson(sample_depth * prob_useful)

                    if this_sample_obs > 0:
                        sample_observations += 1
                        total_read_observations += this_sample_obs

                    # DISCOVERY MODE: Check if we have enough reads for genome coverage
                    # Need reads from at least 2 samples AND enough total reads for coverage
                    if (
                        sample_observations >= self.params["min_sample_observations"]
                        and total_read_observations >= reads_for_coverage
                    ):
                        incidence_at_detection = (
                            cumulative_incidence * processing_delay_factor
                        )
                        return {
                            "cum_incidence": float(incidence_at_detection),
                            "doubling_time": float(doubling_time),
                        }

            if cumulative_incidence > MAX_SUPPORTED_CUMULATIVE_INCIDENCE or day > 365 * 10:
                return {
                    "cum_incidence": 1.0,
                    "doubling_time": float(doubling_time),
                }

    def run_simulations(self, n: int, progress_callback=None) -> List[Dict[str, float]]:
        outcomes = []
        for i in range(n):
            if progress_callback and i % 100 == 0:
                progress_callback(i / n)
            outcomes.append(self.simulate_one())
        if progress_callback:
            progress_callback(1.0)
        return outcomes


def analyze_outcomes(outcomes: List[Dict], p_bad: float, t_gov: float) -> Dict:
    percentiles = [25, 50, 75, 90, 95, 99]
    arr_inc = np.array([o["cum_incidence"] for o in outcomes], dtype=float)
    arr_dt = np.array([o["doubling_time"] for o in outcomes], dtype=float)

    results = {"percentiles": {}}
    results["mean_incidence"] = float(np.mean(arr_inc))
    results["std_incidence"] = float(np.std(arr_inc))
    results["detection_rate"] = float(np.mean(arr_inc < MAX_SUPPORTED_CUMULATIVE_INCIDENCE))

    for p in percentiles:
        results["percentiles"][p] = float(np.percentile(arr_inc, p))

    below_pbad = arr_inc < p_bad
    results["fraction_detected_before_p_bad"] = float(np.mean(below_pbad))

    slack_stats = {}
    if np.any(below_pbad):
        inc_valid = arr_inc[below_pbad]
        dt_valid = arr_dt[below_pbad]
        doublings_to_pbad = np.log(p_bad / inc_valid) / np.log(2.0)
        time_slack_days = doublings_to_pbad * dt_valid

        slack_stats["doublings_percentiles"] = {}
        slack_stats["time_slack_percentiles_days"] = {}
        for p in percentiles:
            slack_stats["doublings_percentiles"][p] = float(np.percentile(doublings_to_pbad, p))
            slack_stats["time_slack_percentiles_days"][p] = float(np.percentile(time_slack_days, p))
        slack_stats["mean_doublings"] = float(np.mean(doublings_to_pbad))
        slack_stats["mean_time_slack_days"] = float(np.mean(time_slack_days))
    else:
        slack_stats["doublings_percentiles"] = None
        slack_stats["time_slack_percentiles_days"] = None
        slack_stats["mean_doublings"] = None
        slack_stats["mean_time_slack_days"] = None

    results["slack_to_p_bad"] = slack_stats

    with np.errstate(divide="ignore", invalid="ignore"):
        doublings_needed = t_gov / arr_dt
        threshold_incidence = p_bad / np.power(2.0, doublings_needed)
    early_enough = arr_inc <= threshold_incidence
    results["fraction_early_enough_for_t_gov"] = float(np.mean(early_enough))

    return results


# Default system configurations
DEFAULT_CURRENT = {
    "name": "Current System",
    "nwss_catchment": 1_000_000,
    "swab_catchment": 191,
    "triturator_catchment": 0,
    "individual_planes_catchment": 0,
    "nwss_delay": 7.0,
    "swab_delay": 5.0,
    "triturator_delay": 5.0,
    "individual_planes_delay": 5.0,
}

DEFAULT_FY2026 = {
    "name": "FY2026 Biothreat Radar",
    "nwss_catchment": 500_000,
    "swab_catchment": 5_200,
    "triturator_catchment": 97_500,
    "individual_planes_catchment": 4_500,
    "nwss_delay": 2.69,
    "swab_delay": 2.19,
    "triturator_delay": 2.65,
    "individual_planes_delay": 2.40,
}

# Fixed parameters (from NAO's model)
FIXED_PARAMS = {
    "cv_doubling_time": 0.1,
    "shedding_values": [
        "Rothman-2697049",  # NWSS wastewater
        [  # Swabs - distribution of relative abundances
            5e-6, 5e-6, 6e-6, 7e-6, 1e-5, 1e-5, 2e-5, 3e-5, 3e-5, 3e-5, 4e-5,
            3e-4, 3e-4, 3e-4, 3e-4, 5e-4, 6e-4, 1e-3, 4e-3, 9e-3, 1e-2, 1e-2,
            2e-2, 3e-2, 4e-2, 5e-2, 5e-2, 6e-2, 2e-1, 2e-1, 3e-1, 3e-1, 4e-1,
            6e-1, 6e-1, 7e-1, 2e-7, 9e-7, 2e-5, 1e-5, 1e-5, 7e-5, 5e-5, 1e-2,
            6e-6, 2e-5, 9e-5, 6e-4, 3e-4, 4e-6, 2e-3, 3e-2, 6e-5, 3e-4, 8e-2,
            2e-4, 2e-4, 2e-4, 1e-6, 3e-5, 2e-4, 1e-5, 3e-5, 1e-3,
        ],
        [1.4e-6],  # Triturators
        [1.4e-6],  # Individual planes
    ],
    "sample_depths": [24e9, 2e9, 188e9, 12e9],
    "sigma_shedding_values": 0.05,
    "shedding_duration": 5.0,
    "sigma_shedding_duration": 0.05,
    "min_sample_observations": 2,  # Still need observations from 2+ samples
    "genome_length_bp": 13_000,  # ~Influenza-sized genome
    "insert_length_bp": 170,
    "fraction_useful_reads": 0.50,
    "target_coverage": 2.0,  # 2x coverage for discovery
}


def build_params(system_config: Dict, doubling_time: float, target_coverage: float) -> Dict:
    """Build full parameter dict from user-editable system config."""
    params = dict(FIXED_PARAMS)
    params["doubling_time"] = doubling_time
    params["target_coverage"] = target_coverage
    params["sample_populations"] = [
        system_config["nwss_catchment"],
        system_config["swab_catchment"],
        system_config["triturator_catchment"],
        system_config["individual_planes_catchment"],
    ]
    params["processing_delays"] = [
        system_config["nwss_delay"],
        system_config["swab_delay"],
        system_config["triturator_delay"],
        system_config["individual_planes_delay"],
    ]
    return params


def format_results(results: Dict, system_name: str) -> str:
    """Format results as readable text."""
    lines = [f"### {system_name}"]
    lines.append("")
    lines.append("**Cumulative incidence at detection:**")
    for p, v in results["percentiles"].items():
        if v < MAX_SUPPORTED_CUMULATIVE_INCIDENCE:
            per_100k = v * 100_000
            infected = POPULATION * v / 1e6
            lines.append(f"- {p}th percentile: **{per_100k:.2f} per 100k** ({infected:.2f}M infected)")
        else:
            lines.append(f"- {p}th percentile: Not detected (>{MAX_SUPPORTED_CUMULATIVE_INCIDENCE:.0%})")
    
    lines.append("")
    lines.append(f"**Detection rate:** {results['detection_rate']:.1%}")
    lines.append(f"**Fraction detected before p_bad:** {results['fraction_detected_before_p_bad']:.1%}")
    lines.append(f"**Fraction early enough for T_gov:** {results['fraction_early_enough_for_t_gov']:.1%}")
    
    slack = results["slack_to_p_bad"]
    if slack["mean_time_slack_days"] is not None:
        lines.append("")
        lines.append(f"**Mean lead time before p_bad:** {slack['mean_time_slack_days']:.1f} days")
        lines.append(f"**Mean doublings of slack:** {slack['mean_doublings']:.1f}")
    
    return "\n".join(lines)


# Streamlit App
st.set_page_config(page_title="Biosurveillance Simulator (Discovery Mode)", layout="wide")

st.title("🦠 Biosurveillance System Comparison - Discovery Mode")

st.markdown("""
This tool compares **discovery** performance (detecting novel/unknown pathogens) of different biosurveillance system configurations.

**Discovery vs Monitoring:**
- **Monitoring**: Detecting known pathogens - just need a few matching reads
- **Discovery**: Characterizing novel pathogens - need ~2x genome coverage to identify what it is

Based on the [NAO's scenario simulator](https://github.com/naobservatory/scenario-simulator).
""")

# Global parameters
st.sidebar.header("Simulation Parameters")
n_simulations = st.sidebar.slider("Number of simulations", 100, 10000, 1000, 100)
doubling_time = st.sidebar.slider("Pathogen doubling time (days)", 1.0, 14.0, 3.0, 0.5)
p_bad = st.sidebar.slider("p_bad (game over threshold)", 0.05, 0.5, 0.3, 0.05)
t_gov = st.sidebar.slider("T_gov (days for govt response)", 14.0, 90.0, 45.0, 1.0)
target_coverage = st.sidebar.slider("Target genome coverage (x)", 1.0, 10.0, 2.0, 0.5)
seed = st.sidebar.number_input("Random seed", value=123, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Key parameters:**
- **p_bad**: Cumulative incidence that's "too late"
- **T_gov**: Days government needs to mount response
- **Doubling time**: How fast pathogen spreads
- **Target coverage**: Genome coverage needed for discovery (2x = characterize novel pathogen)
""")

# Calculate reads needed for coverage
genome_length = FIXED_PARAMS["genome_length_bp"]
insert_length = FIXED_PARAMS["insert_length_bp"]
reads_needed = int(target_coverage * genome_length / insert_length)
st.sidebar.info(f"Reads needed for {target_coverage}x coverage of {genome_length:,}bp genome: **{reads_needed:,}**")

# Two columns for the two systems
col1, col2 = st.columns(2)

with col1:
    st.header("System A")
    name_a = st.text_input("System A Name", DEFAULT_CURRENT["name"], key="name_a")
    
    st.subheader("Catchment Sizes")
    nwss_a = st.number_input("NWSS (wastewater)", 0, 10_000_000, DEFAULT_CURRENT["nwss_catchment"], 100_000, key="nwss_a")
    swab_a = st.number_input("Nasal Swabs", 0, 100_000, DEFAULT_CURRENT["swab_catchment"], 100, key="swab_a")
    trit_a = st.number_input("Triturators", 0, 500_000, DEFAULT_CURRENT["triturator_catchment"], 1_000, key="trit_a")
    plane_a = st.number_input("Individual Planes", 0, 50_000, DEFAULT_CURRENT["individual_planes_catchment"], 500, key="plane_a")
    
    st.subheader("Processing Delays (days)")
    delay_nwss_a = st.number_input("NWSS delay", 0.5, 14.0, DEFAULT_CURRENT["nwss_delay"], 0.5, key="delay_nwss_a")
    delay_swab_a = st.number_input("Swab delay", 0.5, 14.0, DEFAULT_CURRENT["swab_delay"], 0.5, key="delay_swab_a")
    delay_trit_a = st.number_input("Triturator delay", 0.5, 14.0, DEFAULT_CURRENT["triturator_delay"], 0.5, key="delay_trit_a")
    delay_plane_a = st.number_input("Ind. Plane delay", 0.5, 14.0, DEFAULT_CURRENT["individual_planes_delay"], 0.5, key="delay_plane_a")

with col2:
    st.header("System B")
    name_b = st.text_input("System B Name", DEFAULT_FY2026["name"], key="name_b")
    
    st.subheader("Catchment Sizes")
    nwss_b = st.number_input("NWSS (wastewater)", 0, 10_000_000, DEFAULT_FY2026["nwss_catchment"], 100_000, key="nwss_b")
    swab_b = st.number_input("Nasal Swabs", 0, 100_000, DEFAULT_FY2026["swab_catchment"], 100, key="swab_b")
    trit_b = st.number_input("Triturators", 0, 500_000, DEFAULT_FY2026["triturator_catchment"], 1_000, key="trit_b")
    plane_b = st.number_input("Individual Planes", 0, 50_000, DEFAULT_FY2026["individual_planes_catchment"], 500, key="plane_b")
    
    st.subheader("Processing Delays (days)")
    delay_nwss_b = st.number_input("NWSS delay", 0.5, 14.0, DEFAULT_FY2026["nwss_delay"], 0.5, key="delay_nwss_b")
    delay_swab_b = st.number_input("Swab delay", 0.5, 14.0, DEFAULT_FY2026["swab_delay"], 0.5, key="delay_swab_b")
    delay_trit_b = st.number_input("Triturator delay", 0.5, 14.0, DEFAULT_FY2026["triturator_delay"], 0.5, key="delay_trit_b")
    delay_plane_b = st.number_input("Ind. Plane delay", 0.5, 14.0, DEFAULT_FY2026["individual_planes_delay"], 0.5, key="delay_plane_b")

# Build system configs
system_a = {
    "name": name_a,
    "nwss_catchment": nwss_a,
    "swab_catchment": swab_a,
    "triturator_catchment": trit_a,
    "individual_planes_catchment": plane_a,
    "nwss_delay": delay_nwss_a,
    "swab_delay": delay_swab_a,
    "triturator_delay": delay_trit_a,
    "individual_planes_delay": delay_plane_a,
}

system_b = {
    "name": name_b,
    "nwss_catchment": nwss_b,
    "swab_catchment": swab_b,
    "triturator_catchment": trit_b,
    "individual_planes_catchment": plane_b,
    "nwss_delay": delay_nwss_b,
    "swab_delay": delay_swab_b,
    "triturator_delay": delay_trit_b,
    "individual_planes_delay": delay_plane_b,
}

# Run simulations
if st.button("🚀 Run Simulation", type="primary"):
    st.markdown("---")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # System A
    status_text.text(f"Running {name_a}...")
    params_a = build_params(system_a, doubling_time, target_coverage)
    sim_a = BiosurveillanceSimulator(params_a, seed=seed)
    outcomes_a = sim_a.run_simulations(n_simulations, progress_callback=lambda p: progress_bar.progress(p * 0.5))
    results_a = analyze_outcomes(outcomes_a, p_bad, t_gov)
    
    # System B
    status_text.text(f"Running {name_b}...")
    params_b = build_params(system_b, doubling_time, target_coverage)
    sim_b = BiosurveillanceSimulator(params_b, seed=seed)
    outcomes_b = sim_b.run_simulations(n_simulations, progress_callback=lambda p: progress_bar.progress(0.5 + p * 0.5))
    results_b = analyze_outcomes(outcomes_b, p_bad, t_gov)
    
    progress_bar.progress(1.0)
    status_text.text("Complete!")
    
    # Display results
    st.markdown("---")
    st.header("Results")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(format_results(results_a, name_a))
    
    with res_col2:
        st.markdown(format_results(results_b, name_b))
    
    # Comparison summary
    st.markdown("---")
    st.header("Comparison Summary")
    
    median_a = results_a["percentiles"][50] * 100_000
    median_b = results_b["percentiles"][50] * 100_000
    
    if median_a > 0 and median_b > 0:
        ratio = median_a / median_b
        if ratio > 1:
            st.success(f"**{name_b}** detects **{ratio:.1f}x earlier** than {name_a} (median: {median_b:.1f} vs {median_a:.1f} per 100k)")
        else:
            st.success(f"**{name_a}** detects **{1/ratio:.1f}x earlier** than {name_b} (median: {median_a:.1f} vs {median_b:.1f} per 100k)")
    
    t_gov_a = results_a["fraction_early_enough_for_t_gov"] * 100
    t_gov_b = results_b["fraction_early_enough_for_t_gov"] * 100
    
    st.info(f"**Early enough for T_gov ({t_gov:.0f} days):** {name_a}: {t_gov_a:.1f}% | {name_b}: {t_gov_b:.1f}%")

st.markdown("---")
st.caption("Based on [NAO's scenario simulator](https://naobservatory.org/blog/simulating-approaches-to-metagenomic-pandemic-identification/). Discovery mode requires ~2x genome coverage vs monitoring which only needs a few reads.")

use serde::{Deserialize, Serialize};

/// Political ideology spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ideology {
    Progressive,
    Moderate,
    Conservative,
}

/// A policy area that proposals can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolicyArea {
    Taxation,
    PublicTransport,
    Education,
    Healthcare,
    Housing,
    Environment,
    PublicSafety,
    Infrastructure,
}

/// A member of the city council.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilMember {
    pub id: u32,
    pub name: String,
    pub ideology: Ideology,
    /// Priority areas this member cares about.
    pub agenda: Vec<PolicyArea>,
}

impl CouncilMember {
    pub fn new(id: u32, name: String, ideology: Ideology, agenda: Vec<PolicyArea>) -> Self {
        Self {
            id,
            name,
            ideology,
            agenda,
        }
    }

    /// How likely this member is to vote yes on a proposal (0.0 to 1.0).
    pub fn support_score(&self, proposal: &PolicyProposal) -> f32 {
        let mut score = 0.5_f32;

        // Agenda alignment bonus
        if self.agenda.contains(&proposal.area) {
            score += 0.2;
        }

        // Ideology-based bias
        match (self.ideology, proposal.spending_bias()) {
            (Ideology::Progressive, SpendingBias::HighSpend) => score += 0.2,
            (Ideology::Progressive, SpendingBias::LowSpend) => score -= 0.1,
            (Ideology::Conservative, SpendingBias::HighSpend) => score -= 0.2,
            (Ideology::Conservative, SpendingBias::LowSpend) => score += 0.15,
            (Ideology::Progressive, SpendingBias::Neutral) => score += 0.05,
            (Ideology::Conservative, SpendingBias::Neutral) => {}
            (Ideology::Moderate, _) => score += 0.05,
        }

        // Cost-benefit ratio
        if proposal.cost > 0.0 {
            let ratio = proposal.estimated_benefit / proposal.cost;
            if ratio > 2.0 {
                score += 0.15;
            } else if ratio < 0.5 {
                score -= 0.15;
            }
        }

        score.clamp(0.0, 1.0)
    }
}

/// Spending bias of a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpendingBias {
    HighSpend,
    LowSpend,
    Neutral,
}

/// A policy proposal to be voted on by the council.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyProposal {
    pub id: u32,
    pub description: String,
    pub area: PolicyArea,
    pub cost: f64,
    pub estimated_benefit: f64,
    pub enacted: bool,
}

impl PolicyProposal {
    pub fn new(
        id: u32,
        description: String,
        area: PolicyArea,
        cost: f64,
        estimated_benefit: f64,
    ) -> Self {
        Self {
            id,
            description,
            area,
            cost,
            estimated_benefit,
            enacted: false,
        }
    }

    /// Classify the spending level of this proposal.
    pub fn spending_bias(&self) -> SpendingBias {
        if self.cost > 50_000.0 {
            SpendingBias::HighSpend
        } else if self.cost < 10_000.0 {
            SpendingBias::LowSpend
        } else {
            SpendingBias::Neutral
        }
    }
}

/// Result of a council vote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResult {
    pub proposal_id: u32,
    pub votes_for: u32,
    pub votes_against: u32,
    pub passed: bool,
}

/// State snapshot for the council to evaluate.
#[derive(Debug, Clone, Default)]
pub struct CityState {
    pub population: u32,
    pub average_satisfaction: f32,
    pub crime_rate: f32,
    pub unemployment_rate: f32,
    pub pollution_level: f32,
    pub budget: f64,
    pub education_coverage: f32,
    pub health_coverage: f32,
    pub transport_coverage: f32,
}

/// The city council manages members, generates proposals, and votes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CityCouncil {
    pub members: Vec<CouncilMember>,
    pub proposals: Vec<PolicyProposal>,
    pub enacted_policies: Vec<PolicyProposal>,
    pub vote_history: Vec<VoteResult>,
    next_proposal_id: u32,
}

impl CityCouncil {
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            proposals: Vec::new(),
            enacted_policies: Vec::new(),
            vote_history: Vec::new(),
            next_proposal_id: 0,
        }
    }

    /// Add a council member.
    pub fn add_member(&mut self, member: CouncilMember) {
        self.members.push(member);
    }

    /// Remove a council member by id.
    pub fn remove_member(&mut self, id: u32) {
        self.members.retain(|m| m.id != id);
    }

    /// Submit a proposal for consideration.
    pub fn submit_proposal(
        &mut self,
        description: String,
        area: PolicyArea,
        cost: f64,
        estimated_benefit: f64,
    ) -> u32 {
        let id = self.next_proposal_id;
        self.next_proposal_id += 1;
        self.proposals
            .push(PolicyProposal::new(id, description, area, cost, estimated_benefit));
        id
    }

    /// Evaluate the current city state and automatically generate relevant proposals.
    pub fn council_tick(&mut self, state: &CityState) {
        // High crime => public safety proposal
        if state.crime_rate > 0.5 {
            self.submit_proposal(
                "Increase police funding to combat rising crime".to_string(),
                PolicyArea::PublicSafety,
                30_000.0,
                60_000.0,
            );
        }

        // High unemployment => job creation
        if state.unemployment_rate > 0.3 {
            self.submit_proposal(
                "Infrastructure investment for job creation".to_string(),
                PolicyArea::Infrastructure,
                80_000.0,
                120_000.0,
            );
        }

        // Low education coverage => education spending
        if state.education_coverage < 0.5 {
            self.submit_proposal(
                "Build new schools and expand education programs".to_string(),
                PolicyArea::Education,
                60_000.0,
                90_000.0,
            );
        }

        // Low health coverage => healthcare spending
        if state.health_coverage < 0.5 {
            self.submit_proposal(
                "Expand healthcare facilities and services".to_string(),
                PolicyArea::Healthcare,
                70_000.0,
                100_000.0,
            );
        }

        // High pollution => environment
        if state.pollution_level > 0.6 {
            self.submit_proposal(
                "Green initiative to reduce pollution".to_string(),
                PolicyArea::Environment,
                40_000.0,
                80_000.0,
            );
        }

        // Low satisfaction + budget available => housing
        if state.average_satisfaction < 0.4 && state.budget > 100_000.0 {
            self.submit_proposal(
                "Affordable housing initiative".to_string(),
                PolicyArea::Housing,
                100_000.0,
                150_000.0,
            );
        }

        // Transport coverage low
        if state.transport_coverage < 0.4 {
            self.submit_proposal(
                "Expand public transport network".to_string(),
                PolicyArea::PublicTransport,
                90_000.0,
                130_000.0,
            );
        }
    }

    /// Vote on a specific proposal. Returns the result.
    pub fn vote(&mut self, proposal_id: u32) -> Option<VoteResult> {
        let proposal = self.proposals.iter().find(|p| p.id == proposal_id)?;

        let mut votes_for = 0u32;
        let mut votes_against = 0u32;

        for member in &self.members {
            let score = member.support_score(proposal);
            if score >= 0.5 {
                votes_for += 1;
            } else {
                votes_against += 1;
            }
        }

        let passed = votes_for > votes_against;
        let result = VoteResult {
            proposal_id,
            votes_for,
            votes_against,
            passed,
        };

        self.vote_history.push(result.clone());

        // Enact if passed
        if passed {
            if let Some(proposal) = self.proposals.iter_mut().find(|p| p.id == proposal_id) {
                proposal.enacted = true;
                self.enacted_policies.push(proposal.clone());
            }
        }

        // Remove from active proposals
        self.proposals.retain(|p| p.id != proposal_id);

        Some(result)
    }

    /// Vote on all pending proposals. Returns all results.
    pub fn vote_all(&mut self) -> Vec<VoteResult> {
        let ids: Vec<u32> = self.proposals.iter().map(|p| p.id).collect();
        let mut results = Vec::new();
        for id in ids {
            if let Some(result) = self.vote(id) {
                results.push(result);
            }
        }
        results
    }

    /// Number of active (unvoted) proposals.
    pub fn pending_count(&self) -> usize {
        self.proposals.len()
    }

    /// Number of enacted policies.
    pub fn enacted_count(&self) -> usize {
        self.enacted_policies.len()
    }
}

impl Default for CityCouncil {
    fn default() -> Self {
        Self::new()
    }
}

#pragma once

#include <cassert>

//
// Genome
//

inline bool NEAT::Genome::HasNode(NodeGeneId nodeId) const
{
    return m_nodeLinks.find(nodeId) != m_nodeLinks.end(); 
}

inline bool NEAT::Genome::HasConnection(InnovationId innovId) const
{
    return m_connectionGenes.find(innovId) != m_connectionGenes.end(); 
}

inline void NEAT::Genome::AddConnection(const ConnectionGene& c)
{
    const InnovationId innovId = c.m_innovId;
    const NodeGeneId inNode = c.m_inNode;
    const NodeGeneId outNode = c.m_outNode;

    assert(!HasConnection(innovId));
    assert(HasNode(inNode));
    assert(HasNode(outNode));

    m_connectionGenes.insert({ innovId, c });
    m_nodeLinks[inNode].m_outgoings.push_back(innovId);
    m_nodeLinks[outNode].m_incomings.push_back(innovId);

    if (c.m_enabled)
    {
        m_nodeLinks[inNode].m_numEnabledOutgoings++;
        m_nodeLinks[outNode].m_numEnabledIncomings++;
    }
}

inline void NEAT::Genome::DisableConnection(InnovationId innovId)
{
    assert(HasConnection(innovId));
    ConnectionGene& c = m_connectionGenes[innovId];
    c.m_enabled = false;
    m_nodeLinks[c.m_inNode].m_numEnabledOutgoings--;
    m_nodeLinks[c.m_outNode].m_numEnabledIncomings--;
}

inline void NEAT::Genome::AddNode(NodeGeneId nodeId)
{
    assert(!HasNode(nodeId));
    m_nodeLinks.insert({ nodeId, Links() });
}

inline int NEAT::Genome::GetNumConnections() const
{
    return (int)m_connectionGenes.size();
}

inline int NEAT::Genome::GetNumNodes() const
{
    return (int)m_nodeLinks.size();
}

inline auto NEAT::Genome::GetIncommingConnections(NodeGeneId nodeId) const -> const std::vector<InnovationId> &
{
    assert(HasNode(nodeId));
    return m_nodeLinks.at(nodeId).m_incomings;
}

inline auto NEAT::Genome::GetOutgoingConnections(NodeGeneId nodeId) const -> const std::vector<InnovationId> &
{
    assert(HasNode(nodeId));
    return m_nodeLinks.at(nodeId).m_outgoings;
}

//
// Species
//

inline bool NEAT::Species::ShouldProtectBest() const
{
    return m_scores.size() >= 3;
}

inline bool NEAT::Species::operator< (const Species& rhs) const
{
    return m_bestScore.m_fitness < rhs.m_bestScore.m_fitness; 
}

//
// NEAT
//

inline auto NEAT::GetCurrentInnovationId() const -> InnovationId
{
    return m_currentInnovationId;
}

inline auto NEAT::AccessGenome(int index) -> Genome&
{
    return (*m_generation.m_genomes)[index];
}

inline auto NEAT::GetGenome(int index) const -> const Genome& 
{
    return (*m_generation.m_genomes)[index];
}

inline int NEAT::GetNumGenomes() const
{
    return (int)(*m_generation.m_genomes).size();
}

inline auto NEAT::SelectRandomeGenome() -> Genome &
{
    RandomIntDistribution<uint32_t> distribution(0, GetNumGenomes() - 1);
    return AccessGenome(distribution(s_randomGenerator));
}

inline auto NEAT::SelectRandomNodeGene(const std::vector<NodeGeneId>& genes) const -> NodeGeneId
{
    RandomIntDistribution<NodeGeneId> distribution(0, genes.size() - 1);
    return genes[distribution(s_randomGenerator)];
}

inline auto NEAT::SelectRandomNodeGene(const Genome& genome) const -> NodeGeneId
{
    RandomIntDistribution<NodeGeneId> distribution(0, genome.m_nodeLinks.size() - 1);
    return std::next(std::begin(genome.m_nodeLinks), distribution(s_randomGenerator))->first;
}

inline auto NEAT::SelectRandomConnectionGene(const std::vector<InnovationId>& genes) const -> InnovationId
{
    RandomIntDistribution<InnovationId> distribution(0, genes.size() - 1);
    return *std::next(std::begin(genes), distribution(s_randomGenerator));
}

inline auto NEAT::SelectRandomConnectionGene(const Genome& genome) const -> InnovationId
{
    RandomIntDistribution<InnovationId> distribution(0, genome.m_connectionGenes.size() - 1);
    return std::next(std::begin(genome.m_connectionGenes), distribution(s_randomGenerator))->second.m_innovId;
}

inline auto NEAT::GetConnectionGene(Genome& genome, InnovationId innovId) const -> ConnectionGene*
{
    return genome.HasConnection(innovId) ? &genome.m_connectionGenes[innovId] : nullptr;
}

inline auto NEAT::GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*
{
    return genome.HasConnection(innovId) ? &genome.m_connectionGenes.at(innovId) : nullptr;
}

inline auto NEAT::GetNodeGene(NodeGeneId nodeGeneId) const -> const NodeGene &
{
    assert(nodeGeneId < (int)m_generation.m_nodeGenes.size());
    return m_generation.m_nodeGenes[nodeGeneId];
}

inline float NEAT::GetRandomWeight() const
{
    return RandomRealDistribution<float>(m_config.m_minimumWeight, m_config.m_maximumWeight)(s_randomGenerator);
}

inline float NEAT::GetRandomProbability() const
{
    return RandomRealDistribution<float>(0.0f, 1.0f)(s_randomGenerator);
}

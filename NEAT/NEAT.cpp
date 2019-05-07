#include "NEAT.h"

#include <random>
#include <bitset>
#include <cassert>
#include <algorithm>
#include <queue>

static std::default_random_engine randomGenerator;
template <typename T>
using RandomIntDistribution = std::uniform_int_distribution<T>;
template <typename T>
using RandomRealDistribution = std::uniform_real_distribution<T>;

// Initialize NEAT
// This has to be called before gaining any generations
// Returns the initial generation
auto NEAT::Initialize(const Configration& configIn) -> const Generation &
{
    config = configIn;

    Reset();

    // Populate the first generation with default genoms
    generation.genomes.reserve(config.numOrganismsInGeneration);
    for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        generation.genomes.push_back(CreateDefaultInitialGenome());
    }

    currentInnovationId = GetNumConnectionsInDefaultGenome();
    currentNodeGeneId = GetNumNodesInDefaultGenome();

    return generation;
}

// Reset all the state in NEAT
// After calling this function, Initialize has to be called in order to run NEAT again
void NEAT::Reset()
{
    generation = Generation();
    currentInnovationId = 0;
    currentNodeGeneId = 0;
}

// Gain the new generation
auto NEAT::GetNewGeneration() -> const Generation&
{
    GenerateNewGeneration();
    return generation;
}

// Add a new node at a random connection
void NEAT::AddNewNode(Genome& genome)
{
    // Collect all connections where we can add new node first
    std::vector<InnovationId> availableConnections;
    {
        availableConnections.reserve(genome.connectionGenes.size());

        for (const auto& gene : genome.connectionGenes)
        {
            if (gene.enabled)
            {
                availableConnections.push_back(gene.innovId);
            }
        }

        // Return when there's no available connection to add a new node
        if (availableConnections.size() == 0)
        {
            // TODO: Output a useful message
            return;
        }
    }

    // Create a new node
    const NodeGeneId newNodeId = currentNodeGeneId++;
    NodeGene newNode{ newNodeId, NodeGeneType::Hidden };
    genome.nodeGenes.push_back(newNode);

    // Choose a random connection gene from the available ones
    InnovationId innovId = SelectRandomConnectionGene(availableConnections);
    ConnectionGene* gene = GetConnectionGene(genome, innovId);
    assert(gene);

    // Disable existing gene
    gene->enabled = false;

    NodeGeneId inNode = gene->inNode;
    NodeGeneId outNode = gene->outNode;

    // Create a new connection between the inNode and the new node
    auto newInnovId1 = currentInnovationId++;
    {
        ConnectionGene newConnection;
        newConnection.innovId = newInnovId1;
        newConnection.inNode = inNode;
        newConnection.outNode = newNodeId;
        newConnection.weight = gene->weight;
        newConnection.enabled = true;
        newConnection.activationFuncId = gene->activationFuncId;
        genome.connectionGenes.push_back(newConnection);
    }

    // Create a new connection between the new node and the outNode
    auto newInnovId2 = currentInnovationId++;
    {
        ConnectionGene newConnection;
        newConnection.innovId = newInnovId2;
        newConnection.inNode = newNodeId;
        newConnection.outNode = outNode;
        newConnection.weight = 1.f;
        newConnection.enabled = true;
        InitActivationFunc(newConnection);
        genome.connectionGenes.push_back(newConnection);
    }

    // Update cached relationships between node genes and connection genes
    // Disabled connection stays in these maps
    genome.outgoingConnectionList[inNode].push_back(newInnovId1);
    genome.incomingConnectionList[outNode].push_back(newInnovId2);
    genome.incomingConnectionList[newNodeId].push_back(newInnovId1);
    genome.outgoingConnectionList[newNodeId].push_back(newInnovId2);

    // Remember the newly added node
    newlyAddedNodes[innovId].push_back(GenomeNodePair(genome, newNodeId));
}

// Add a new connection between random two nodes
void NEAT::AddNewConnection(Genome& genome)
{
    if (config.allowCyclicNetwork)
    {
        AddNewConnectionAllowingCyclic(genome);
    }
    else
    {
        AddNewForwardConnection(genome);
    }
}

// Add a new connection between random two nodes allowing cyclic network
void NEAT::AddNewConnectionAllowingCyclic(Genome& genome)
{

    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> availableNodes;
    {
        const size_t numNodes = genome.nodeGenes.size();
        for (const auto& node : genome.nodeGenes)
        {
            if (genome.outgoingConnectionList[node.id].size() < numNodes)
            {
                availableNodes.push_back(node.id);
            }
        }

        // Return when there's no available node to add a new connection
        if (availableNodes.size() == 0)
        {
            // TODO: Output a useful message
            return;
        }
    }

    // Select a random node from the available ones as inNode
    const NodeGeneId node1 = SelectRandomNodeGene(availableNodes);

    // Then collect all node genes which are not connected to the inNode
    {
        availableNodes.clear();

        std::vector<int> connectedFlag;
        // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
        connectedFlag.resize(currentNodeGeneId / sizeof(int) + 1, 0);

        for (auto innovId : genome.outgoingConnectionList[node1])
        {
            const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
            assert(cgene);
            NodeGeneId nodeId = cgene->outNode;
            int index = nodeId / sizeof(int);
            connectedFlag[index] = connectedFlag[index] | (1 << (nodeId % sizeof(int)));
        }

        // Find all nodes not connected to node1
        for (const auto& node : genome.nodeGenes)
        {
            if (((connectedFlag[node.id / sizeof(int)] >> (node.id % sizeof(int))) & 1) == 0)
            {
                availableNodes.push_back((NodeGeneId)node.id);
            }
        }
    }

    // availableNodes must not empty here because node1 has at least one other node which it's not connected to
    assert(availableNodes.size() > 0);
    // Select a random node from the available ones as outNode
    NodeGeneId node2 = SelectRandomNodeGene(availableNodes);

    // Easier but not reliable way. It could be faster when there are only a few connections.
    //do
    //{
    //    node1 = SelectRandomNodeGene(genome);
    //    node2 = SelectRandomNodeGene(genome);
    //}
    //while (node1 == node2 || ConnectionExists(genome, node1, node2));

    // Add a new connection
    ConnectionGene newConnection;
    auto innovId = currentInnovationId++;
    newConnection.innovId = innovId;
    newConnection.inNode = node1;
    newConnection.outNode = node2;
    RandomRealDistribution<float> floatDistribution(0, 1.f);
    newConnection.weight = floatDistribution(randomGenerator);
    newConnection.enabled = true;
    InitActivationFunc(newConnection);
    genome.connectionGenes.push_back(newConnection);

    // Update relationship between nodes and connections
    genome.outgoingConnectionList[node1].push_back(innovId);
    genome.incomingConnectionList[node2].push_back(innovId);

    // Remember the newly added connection
    newlyAddedConnections[NodePair(node1, node2)].push_back(GenomeConnectionPair(genome, innovId));
}

// Add a new connection between random two nodes without allowing cyclic network
// Direction of the new connection is guaranteed to be one direction (distance from in node to an input node is smaller than the one of out node)
void NEAT::AddNewForwardConnection(Genome& genome)
{
    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> availableInNodes;
    {
        const size_t numNodes = genome.nodeGenes.size();
        for (const auto& node : genome.nodeGenes)
        {
            if (genome.outgoingConnectionList[node.id].size() < numNodes)
            {
                availableInNodes.push_back(node.id);
            }
        }

        // Return when there's no available node to add a new connection
        if (availableInNodes.size() == 0)
        {
            // TODO: Output a useful message message
            return;
        }
    }

    // Randomize the order of available nodes in the array
    std::random_shuffle(availableInNodes.begin(), availableInNodes.end());

    // Find a node pair where node1 has smaller distance to an input node than the one of node2
    NodeGeneId node1 = invalidGeneId;
    NodeGeneId node2 = invalidGeneId;
    for (auto node : availableInNodes)
    {
        const int node1Depth = GetNodeDepth(genome, node);

        std::vector<NodeGeneId> availableOutNodes;

        std::vector<int> connectedFlag;
        // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
        connectedFlag.resize(currentNodeGeneId / sizeof(int) + 1, 0);

        for (auto innovId : genome.outgoingConnectionList[node])
        {
            const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
            assert(cgene);
            NodeGeneId nodeId = cgene->outNode;
            int index = nodeId / sizeof(int);
            connectedFlag[index] = connectedFlag[index] | (1 << (nodeId % sizeof(int)));
        }

        for (const auto& node : genome.nodeGenes)
        {
            if (((connectedFlag[node.id / sizeof(int)] >> (node.id % sizeof(int))) & 1) == 0)
            {
                // outNode has to have larger depth than inNode in order to prevent circular network
                if (GetNodeDepth(genome, node.id) > node1Depth)
                {
                    availableOutNodes.push_back(node.id);
                }
            }
        }

        if (availableOutNodes.size() == 0)
        {
            // No available node where we can create a forward connection found
            continue;
        }

        node1 = node;
        // Select a random node from the available ones as outNode
        node2 = SelectRandomNodeGene(availableOutNodes);

        // TODO: Come up with a better solution here.
        // We should be able to simply swap node1 and node2 when node2 has a bigger depth.
    }

    // Return when there's no available nodes
    if (node1 == invalidGeneId)
    {
        // TODO: Output a useful message
        return;
    }

    // Add a new connection
    ConnectionGene newConnection;
    auto innovId = currentInnovationId++;
    newConnection.innovId = innovId;
    newConnection.inNode = node1;
    newConnection.outNode = node2;
    RandomRealDistribution<float> floatDistribution(0, 1.f);
    newConnection.weight = floatDistribution(randomGenerator);
    newConnection.enabled = true;
    InitActivationFunc(newConnection);
    genome.connectionGenes.push_back(newConnection);

    // Update relationship between nodes and connections
    genome.outgoingConnectionList[node1].push_back(innovId);
    genome.incomingConnectionList[node2].push_back(innovId);

    // Remember the newly added connection
    newlyAddedConnections[NodePair(node1, node2)].push_back(GenomeConnectionPair(genome, innovId));
}

// Get shortest distance from a node to an input node
// This function has to be called only when allowCyclicNetwork is false
int NEAT::GetNodeDepth(Genome& genome, NodeGeneId id) const
{
    assert(config.allowCyclicNetwork == false);

    // Breadth search input node
    using NodeAndDepth = std::pair<NodeGeneId, int>;
    std::queue<NodeAndDepth> q;

    // Collect all parent nodes of the input node
    for (const auto& iid : genome.incomingConnectionList[id])
    {
        const auto cn = GetConnectionGene(genome, iid);
        assert(cn);
        q.push(NodeAndDepth(cn->inNode, 1));
    }

    // Find the nearest input node
    while (!q.empty())
    {
        auto e = q.front();
        q.pop();

        const auto n = GetNodeGene(genome, e.first);
        assert(n);
        if (n->type == NodeGeneType::Input)
        {
            return e.second;
        }

        // Add parent nodes to the queue
        for (const auto& iid : genome.incomingConnectionList[e.first])
        {
            const auto cn = GetConnectionGene(genome, iid);
            assert(cn);
            q.push(NodeAndDepth(cn->inNode, e.second + 1));
        }
    }

    // Couldn't find input node
    return -1;
}

// Perform cross over operation over two genomes and generate a new genome
// This function assumes that genome1 has a higher fitness value than genome2
// Set sameFitness true when genome1 and genome2 have the same fitness values
auto NEAT::CrossOver(const Genome& parent1, const Genome& parent2, bool sameFitness) const -> Genome
{
    // Create a new genome
    Genome child;

    // Inherite connection genes from the two parents
    {
        RandomIntDistribution<int> random(0, 1);
        RandomRealDistribution<float> randomf(0, 1.f);

        const auto& cGenes1 = parent1.connectionGenes;
        const auto& cGenes2 = parent2.connectionGenes;
        const size_t numConnectionsP1 = cGenes1.size();
        const size_t numConnectionsP2 = cGenes2.size();
        size_t ip1, ip2;
        bool inheriteFromP1;

        // Select each gene from either parent based on gene's innovation id and parents' fitness
        // Note that cGenes1/2 are sorted by innovation id
        for (ip1 = 0, ip2 = 0; ip1 < numConnectionsP1 && ip2 < numConnectionsP2;)
        {
            inheriteFromP1 = true;
            const ConnectionGene& cGene1 = cGenes1[ip1];
            const ConnectionGene& cGene2 = cGenes2[ip2];

            bool enabled = true;

            // When two genes have the same innovation id, take one gene randomly from either parent
            // Or if two genes have the same fittness values, always take one gene randomly from either parent regardless of innovation id
            bool sameInnovation = cGene1.innovId == cGene2.innovId;
            if (sameInnovation || sameFitness)
            {
                if (random(randomGenerator))
                {
                    inheriteFromP1 = false;
                }

                if (sameInnovation)
                {
                    // The gene of the new genome could be disable when the gene of either parent is disabled
                    if (!cGene1.enabled || !cGene2.enabled)
                    {
                        if (randomf(randomGenerator) < config.geneDisablingRate)
                        {
                            enabled = false;
                        }
                    }

                    ++ip1;
                    ++ip2;
                }
                else
                {
                    cGene1.innovId < cGene2.innovId ? ++ip1 : ++ip2;
                }
            }
            // If this gene exists only in parent1, inherite from parent1
            else if (cGene1.innovId < cGene2.innovId)
            {
                ++ip1;
            }
            // If this gene existss only in parent2, don't inherite it
            else
            {
                ++ip2;
                continue;
            }

            // Add connection gene to the new child
            if (inheriteFromP1 && cGene1.innovId <= cGene2.innovId)
            {
                child.connectionGenes.push_back(cGene1);
            }
            else if (!inheriteFromP1 && cGene2.innovId <= cGene1.innovId)
            {
                child.connectionGenes.push_back(cGene2);
            }
            else
            {
                assert(0); // This shouldn't happen
            }

            child.connectionGenes.back().enabled = enabled;
        }

        // Add remaining genes
        while (ip1 < numConnectionsP1)
        {
            if (!sameFitness || random(randomGenerator))
            {
                child.connectionGenes.push_back(cGenes1[ip1]);
            }
            ++ip1;
        }
        while (ip2 < numConnectionsP2)
        {
            if (sameFitness && random(randomGenerator))
            {
                child.connectionGenes.push_back(cGenes2[ip2]);
            }
            ++ip2;
        }
    }

    // Add node genes to the child genome
    {
        std::unordered_map<NodeGeneId, NodeGene> nodeGenes;
        for (const auto& connectionGene : child.connectionGenes)
        {
            if (nodeGenes.find(connectionGene.inNode) == nodeGenes.end())
            {
                NodeGene newNode{ connectionGene.inNode, NodeGeneType::Hidden };
                nodeGenes[connectionGene.inNode] = newNode;
            }

            if (nodeGenes.find(connectionGene.outNode) == nodeGenes.end())
            {
                NodeGene newNode{ connectionGene.outNode, NodeGeneType::Hidden };
                nodeGenes[connectionGene.outNode] = newNode;
            }
        }

        child.nodeGenes.reserve(nodeGenes.size());

        for (const auto& elem : nodeGenes)
        {
            child.nodeGenes.push_back(elem.second);
        }
    }

    // Populate node connection map
    for (const auto& connectionGene : child.connectionGenes)
    {
        child.incomingConnectionList[connectionGene.outNode].push_back(connectionGene.innovId);
        child.outgoingConnectionList[connectionGene.inNode].push_back(connectionGene.innovId);
    }

    return child;
}

// Implementation of generating a new generation
void NEAT::GenerateNewGeneration()
{
    RandomRealDistribution<float> randomf(0, 1.f);

    // Weight mutation
    for (auto& genome : generation.genomes)
    {
        for (auto& connection : genome.connectionGenes)
        {
            if (randomf(randomGenerator) <= config.weightMutationRate)
            {
                if (randomf(randomGenerator) <= config.weightPerturbationRate)
                {
                    connection.weight *= 1.f + (randomf(randomGenerator) - 0.5f) * 0.025f;
                }
                else
                {
                    connection.weight = randomf(randomGenerator);
                }
            }
        }
    }

    // Topological change
    for (auto& genome : generation.genomes)
    {
        if (randomf(randomGenerator) <= config.nodeAdditionRate)
        {
            AddNewNode(genome);
        }
        if (randomf(randomGenerator) <= config.connectionAdditionRate)
        {
            AddNewConnection(genome);
        }
    }

    // Make sure that the same topological changes have the same id
    {
        for (auto& elem : newlyAddedNodes)
        {
            auto& genomes = elem.second;
            NodeGeneId id = genomes[0].second;
            for (size_t i = 1; i < genomes.size(); ++i)
            {
                NodeGeneId thisId = genomes[i].second;
                Genome& genome = genomes[i].first;
                for (auto& node : genome.nodeGenes)
                {
                    if (node.id == thisId)
                    {
                        node.id = id;
                    }
                }
                for (auto& connection : genome.connectionGenes)
                {
                    if(connection.inNode == thisId)
                    {
                        connection.inNode = id;
                    }
                    if (connection.outNode == thisId)
                    {
                        connection.outNode = id;
                    }
                }
                assert(genome.incomingConnectionList.find(id) == genome.incomingConnectionList.end());
                assert(genome.outgoingConnectionList.find(id) == genome.outgoingConnectionList.end());
                genome.incomingConnectionList[id] = genome.incomingConnectionList[thisId];
                genome.incomingConnectionList.erase(thisId);
                genome.outgoingConnectionList[id] = genome.outgoingConnectionList[thisId];
                genome.outgoingConnectionList.erase(thisId);
            }
        }
        for (auto& elem : newlyAddedConnections)
        {
            auto& genomes = elem.second;
            InnovationId id = genomes[0].second;
            for (size_t i = 1; i < genomes.size(); ++i)
            {
                InnovationId thisId = genomes[i].second;
                Genome& genome = genomes[i].first;
                for (auto& connection : genome.connectionGenes)
                {
                    if (connection.innovId == thisId)
                    {
                        connection.innovId = id;
                    }
                }
                for (auto& ids : genome.incomingConnectionList)
                {
                    for (size_t j = 0; j < ids.second.size(); ++j)
                    {
                        if (ids.second[j] == thisId)
                        {
                            ids.second[j] = id;
                        }
                    }
                }
                for (auto& ids : genome.outgoingConnectionList)
                {
                    for (size_t j = 0; j < ids.second.size(); ++j)
                    {
                        if (ids.second[j] == thisId)
                        {
                            ids.second[j] = id;
                        }
                    }
                }
            }
        }

        newlyAddedNodes.clear();
        newlyAddedConnections.clear();
    }

    const int numCopiedOrgs = int(config.numOrganismsInGeneration * (1.f - config.crossOverRate));

    // Copy the current generation
    Generation curGen = generation;

    // Evaluate genomes and copy high score organizations to the next gen
    struct Score
    {
        float fitness;
        int index;
        bool operator< (const Score& rhs) { return fitness < rhs.fitness; }
    };

    std::vector<Score> scores;
    scores.reserve(config.numOrganismsInGeneration);
    for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        Score score;
        score.fitness = Evaluate(curGen.genomes[i]);
        score.index = i;
        scores.push_back(score);
    }

    std::sort(scores.begin(), scores.end());

    for (int i = 0; i < numCopiedOrgs; ++i)
    {
        generation.genomes[i] = curGen.genomes[scores[i].index];
    }

    // Cross Over
    RandomIntDistribution<int> randomi(0, config.numOrganismsInGeneration - 1);
    for (uint32_t i = numCopiedOrgs; i < config.numOrganismsInGeneration; ++i)
    {
        int i1 = randomi(randomGenerator);
        int i2 = randomi(randomGenerator);
        while (i1 == i2)
        {
            i2 = randomi(randomGenerator);
        }
        if (i1 < i2)
        {
            std::swap(i1, i2);
        }
        generation.genomes[i] = CrossOver(
            curGen.genomes[scores[i1].index],
            curGen.genomes[scores[i2].index],
            scores[i1].fitness == scores[i2].fitness);
    }

    generation.generationId = curGen.generationId + 1;
}

float NEAT::Evaluate(const Genome& genom) const
{
    if (fitnessFunc)
    {
        return fitnessFunc(genom);
    }

    return 0.0f;
}

auto NEAT::CreateDefaultInitialGenome() const -> Genome
{
    Genome genom;

    genom.nodeGenes.reserve(3);
    auto& nodeGenes = genom.nodeGenes;

    // Create an input node
    {
        NodeGene node = { 0, NodeGeneType::Input };
        nodeGenes.push_back(node);
    }

    // Create a hidden node
    {
        NodeGene node = { 1, NodeGeneType::Hidden };
        nodeGenes.push_back(node);
    }

    // Create an output node
    {
        NodeGene node = { 2, NodeGeneType::Output };
        nodeGenes.push_back(node);
    }

    genom.connectionGenes.reserve(2);
    auto& connections = genom.connectionGenes;

    // Create a connection between the input and the hidden node
    {
        // InnovationId of this connection for default init genom will be always 0
        connections.push_back(ConnectionGene{ 0, 0, 1, defaultActivationFuncId, 1.f, true });
        genom.outgoingConnectionList[0].push_back(0);
        genom.incomingConnectionList[1].push_back(0);
    }

    // Create a connection between the hidden and the output node
    {
        // InnovationId of this connection for default init genom will be always 1
        connections.push_back(ConnectionGene{ 1, 1, 2, defaultActivationFuncId, 1.f, true });
        genom.outgoingConnectionList[1].push_back(1);
        genom.incomingConnectionList[2].push_back(1);
    }

    return genom;
}

int NEAT::GetNumConnectionsInDefaultGenome() const
{
    return 2;
}

int NEAT::GetNumNodesInDefaultGenome() const
{
    return 3;
}

auto NEAT::SelectRandomeGenome() -> Genome&
{
    RandomIntDistribution<uint32_t> distribution(0, generation.genomes.size() - 1);
    return generation.genomes[distribution(randomGenerator)];
}

auto NEAT::SelectRandomNodeGene(const std::vector<NodeGeneId>& genes) const -> NodeGeneId
{
    RandomIntDistribution<NodeGeneId> distribution(0, genes.size() - 1);
    return genes[distribution(randomGenerator)];
}

auto NEAT::SelectRandomNodeGene(const Genome& genome) const -> NodeGeneId
{
    RandomIntDistribution<NodeGeneId> distribution(0, genome.nodeGenes.size() - 1);
    return genome.nodeGenes[distribution(randomGenerator)].id;
}

auto NEAT::SelectRandomConnectionGene(const std::vector<InnovationId>& genes) const -> InnovationId
{
    RandomIntDistribution<InnovationId> distribution(0, genes.size() - 1);
    return genes[distribution(randomGenerator)];
}

auto NEAT::SelectRandomConnectionGene(const Genome& genome) const -> InnovationId
{
    RandomIntDistribution<InnovationId> distribution(0, genome.connectionGenes.size() - 1);
    return  genome.connectionGenes[distribution(randomGenerator)].innovId;
}

void NEAT::InitActivationFunc(ConnectionGene& gene) const
{
    if (config.useGlobalActivationFunc)
    {
        gene.activationFuncId = defaultActivationFuncId;
    }
    else
    {
        // Set a random activation func
        RandomIntDistribution<uint32_t> distribution(0, activationFuncs.size());
        gene.activationFuncId = distribution(randomGenerator);
    }
}

auto NEAT::GetConnectionGene(Genome& genome, InnovationId innovId) -> ConnectionGene*
{
    int index = FindGeneBinarySearch(
        genome.connectionGenes,
        innovId,
        [](const ConnectionGene& gene)
        {
            return gene.innovId;
        });
    if (index < 0) return nullptr;
    return &genome.connectionGenes[index];
}

auto NEAT::GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*
{
    int index = FindGeneBinarySearch(
        genome.connectionGenes,
        innovId,
        [](const ConnectionGene& gene)
        {
            return gene.innovId;
        });
    if (index < 0) return nullptr;
    return &genome.connectionGenes[index];
}

auto NEAT::GetNodeGene(const Genome& genome, NodeGeneId id) const -> const NodeGene*
{
    int index = FindGeneBinarySearch(
        genome.nodeGenes,
        id,
        [](const NodeGene & gene)
        {
            return gene.id;
        });
    if (index < 0) return nullptr;
    return &genome.nodeGenes[index];
}

template <typename Gene, typename FuncType>
int NEAT::FindGeneBinarySearch(const std::vector<Gene>& genes, uint32_t id, FuncType getIdFunc)
{
    int low = 0;
    int high = genes.size() - 1;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (getIdFunc(genes[mid]) == id)
        {
            return mid;
        }
        else if (getIdFunc(genes[mid]) > id)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }

    return -1;
}
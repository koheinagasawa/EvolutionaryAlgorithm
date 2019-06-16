#include "NEAT.h"

#include <random>
#include <bitset>
#include <cassert>
#include <algorithm>
#include <stack>
#include <iostream>

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
    isInitialized = false;

    config = configIn;

    Reset();

    if (config.numOrganismsInGeneration == 0)
    {
        // TODO: Add some useful warning message here
        return generation;
    }

    // Populate the first generation with default genoms
    generation.genomes.reserve(config.numOrganismsInGeneration);
    for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        generation.genomes.push_back(CreateDefaultInitialGenome());
    }

    {
        const Genome& g = generation.genomes[0];
        currentInnovationId = g.connectionGenes.size();
        currentNodeGeneId = g.nodeGenes.size();
    }

    if (config.activateFunctions.size() == 0 || config.activateFunctions.size() <= config.defaultActivateFunctionId)
    {
        // TODO: Add some useful warning message here
        return generation;
    }

    activationFuncs = config.activateFunctions;
    defaultActivationFuncId = config.defaultActivateFunctionId;

    isInitialized = true;

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
    if (isInitialized)
    {
        GenerateNewGeneration();
    }
    return generation;
}

// Return the current generation
auto NEAT::GetCurrentGeneration() -> const Generation&
{
    return generation;
}

// Print fitness of each genome in the current generation
void NEAT::PrintFitness() const
{
    std::cout << "\nGeneration " << generation.generationId << std::endl;
    float sum = 0;
    const int size = generation.genomes.size();
    float maxFitness = 0;
    int bestGenomeId = -1;
    for (int i = 0; i < size; ++i)
    {
        const Genome& genome = generation.genomes[i];
        const float fitness = Evaluate(genome);
        //std::cout << "Genome " << i << ": " << fitness << std::endl;
        sum += fitness;
        if (fitness > maxFitness)
        {
            maxFitness = fitness;
            bestGenomeId = i;
        }
    }
    std::cout << "Average fitness over " << size << " organims: " << sum / (float)size << std::endl;
    std::cout << "Maximum fitness: Genome " << bestGenomeId << " - " << maxFitness << std::endl;
}


auto NEAT::GetNewNodeGeneId() const -> NodeGeneId
{
    // TODO: Make it thread safe
    return currentNodeGeneId++;
}

auto NEAT::GetNewInnovationId() const -> InnovationId
{
    // TODO: Make it thread safe
    return currentInnovationId++;
}

// Add a new node at a random connection
auto NEAT::AddNewNode(Genome& genome) const -> NewlyAddedNode
{
    // Collect all connections where we can add new node first
    std::vector<InnovationId> availableConnections;
    {
        availableConnections.reserve(genome.connectionGenes.size());

        for (const ConnectionGene& gene : genome.connectionGenes)
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
            return NewlyAddedNode{genome, invalidGenerationId, invalidInnovationId};
        }
    }

    // Create a new node
    NodeGene newNode = GenerateNewNode();
    const NodeGeneId newNodeId = newNode.id;
    genome.nodeGenes.push_back(newNode);

    // Choose a random connection gene from the available ones
    const InnovationId innovId = SelectRandomConnectionGene(availableConnections);
    ConnectionGene* gene = GetConnectionGene(genome, innovId);
    assert(gene);

    // Disable existing gene
    gene->enabled = false;

    // Get values of this connection gene before we modify the genome
    // (the pointer to gene can become invalid after making any modification to the genome)
    const NodeGeneId inNode = gene->inNode;
    const NodeGeneId outNode = gene->outNode;
    const float weight = gene->weight;

    // Create a new connection between the inNode and the new node
    Connect(inNode, newNodeId, weight, genome);

    // Create a new connection between the new node and the outNode
    Connect(newNodeId, outNode, 1.f, genome);

    return NewlyAddedNode{ genome, newNodeId, innovId };
}

auto NEAT::GenerateNewNode() const -> NodeGene
{
    NodeGene node{ GetNewNodeGeneId(), NodeGeneType::Hidden };

    if (config.useGlobalActivationFunc)
    {
        node.activationFuncId = defaultActivationFuncId;
    }
    else
    {
        // Set a random activation func
        RandomIntDistribution<uint32_t> distribution(0, activationFuncs.size());
        node.activationFuncId = distribution(randomGenerator);
    }
    return node;
}

auto NEAT::Connect(NodeGeneId inNode, NodeGeneId outNode, float weight, Genome& genome) const -> InnovationId
{
    assert(GetNodeGene(genome, inNode));
    assert(GetNodeGene(genome, outNode));

    ConnectionGene gene;
    gene.innovId = GetNewInnovationId();
    gene.inNode = inNode;
    gene.outNode = outNode;
    gene.enabled = true;
    gene.weight = weight;
    genome.connectionGenes.push_back(gene);
    genome.incomingConnectionList[outNode].push_back(gene.innovId);
    genome.outgoingConnectionList[inNode].push_back(gene.innovId);

    return gene.innovId;
}

// Add a new connection between random two nodes
auto NEAT::AddNewConnection(Genome& genome) const -> NewlyAddedConnection
{
    if (config.allowCyclicNetwork)
    {
        return AddNewConnectionAllowingCyclic(genome);
    }
    else
    {
        return AddNewForwardConnection(genome);
    }
}

// Add a new connection between random two nodes allowing cyclic network
auto NEAT::AddNewConnectionAllowingCyclic(Genome& genome) const -> NewlyAddedConnection
{
    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> availableNodes;
    {
        const size_t numNodes = genome.nodeGenes.size();
        for (const NodeGene& node : genome.nodeGenes)
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
            return NewlyAddedConnection{ genome, invalidInnovationId, NodePair() };
        }
    }

    // Select a random node from the available ones as inNode
    const NodeGeneId node1 = SelectRandomNodeGene(availableNodes);

    // Then collect all node genes which are not connected to the inNode
    {
        availableNodes.clear();

        std::vector<int> connectedFlag;
        // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
        connectedFlag.resize((GetCurrentNodeGeneId() / sizeof(int)) + 1, 0);

        for (auto innovId : genome.outgoingConnectionList.at(node1))
        {
            const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
            assert(cgene);
            const NodeGeneId nodeId = cgene->outNode;
            connectedFlag[nodeId / sizeof(int)] |= (1 << (nodeId % sizeof(int)));
        }

        // Find all nodes not connected to node1
        for (const NodeGene& node : genome.nodeGenes)
        {
            if (node.type != NodeGeneType::Hidden && node.type != NodeGeneType::Output)
            {
                continue;
            }

            if (SameConnectionExist(genome, node1, node.id))
            {
                continue;
            }

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
    auto innovId = GetNewInnovationId();
    newConnection.innovId = innovId;
    newConnection.inNode = node1;
    newConnection.outNode = node2;
    RandomRealDistribution<float> floatDistribution(-1.f, 1.f);
    newConnection.weight = floatDistribution(randomGenerator);
    newConnection.enabled = true;
    genome.connectionGenes.push_back(newConnection);

    // Update relationship between nodes and connections
    genome.outgoingConnectionList[node1].push_back(innovId);
    genome.incomingConnectionList[node2].push_back(innovId);

#ifdef _DEBUG
    CheckSanity(genome);
#endif

    return NewlyAddedConnection{ genome, innovId, NodePair{node1, node2} };
}

// Add a new connection between random two nodes without allowing cyclic network
// Direction of the new connection is guaranteed to be one direction (distance from in node to an input node is smaller than the one of out node)
auto NEAT::AddNewForwardConnection(Genome& genome) const -> NewlyAddedConnection
{
    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> availableInNodes;
    {
        const size_t numNodes = genome.nodeGenes.size();
        for (const auto& node : genome.nodeGenes)
        {
            if (node.type != NodeGeneType::Output &&
                genome.outgoingConnectionList[node.id].size() < numNodes)
            {
                availableInNodes.push_back(node.id);
            }
        }

        // Return when there's no available node to add a new connection
        if (availableInNodes.size() == 0)
        {
            // TODO: Output a useful message message
            return NewlyAddedConnection{ genome, invalidInnovationId, NodePair() };
        }
    }

    // Randomize the order of available nodes in the array
    std::random_shuffle(availableInNodes.begin(), availableInNodes.end());

    NodeGeneId node1 = invalidGeneId;
    NodeGeneId node2 = invalidGeneId;
    for (auto node : availableInNodes)
    {
        std::vector<NodeGeneId> availableOutNodes;

        std::vector<int> connectedFlag;
        // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
        connectedFlag.resize(GetCurrentNodeGeneId() / sizeof(int) + 1, 0);

        for (auto innovId : genome.outgoingConnectionList[node])
        {
            const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
            assert(cgene);
            const NodeGeneId nodeId = cgene->outNode;
            connectedFlag[nodeId / sizeof(int)] |= (1 << (nodeId % sizeof(int)));
        }

        for (const auto& n : genome.nodeGenes)
        {
            if (n.type != NodeGeneType::Hidden && n.type != NodeGeneType::Output)
            {
                continue;
            }

            if (SameConnectionExist(genome, node1, n.id))
            {
                continue;
            }

            if (((connectedFlag[n.id / sizeof(int)] >> (n.id % sizeof(int))) & 1) == 0)
            {
                if (CheckCyclic(genome, node, n.id))
                {
                    availableOutNodes.push_back(n.id);
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
        return NewlyAddedConnection{ genome, invalidInnovationId, NodePair() };
    }

    // Add a new connection
    ConnectionGene newConnection;
    auto innovId = GetNewInnovationId();
    newConnection.innovId = innovId;
    newConnection.inNode = node1;
    newConnection.outNode = node2;
    RandomRealDistribution<float> floatDistribution(-1.f, 1.f);
    newConnection.weight = floatDistribution(randomGenerator);
    newConnection.enabled = true;
    genome.connectionGenes.push_back(newConnection);

    // Update relationship between nodes and connections
    genome.outgoingConnectionList[node1].push_back(innovId);
    genome.incomingConnectionList[node2].push_back(innovId);

#ifdef _DEBUG
    CheckSanity(genome);
#endif

    return NewlyAddedConnection{ genome, innovId, NodePair{node1, node2} };
}

bool NEAT::CheckCyclic(const Genome& genome, NodeGeneId srcNode, NodeGeneId targetNode) const
{
    if (srcNode == targetNode)
    {
        return false;
    }

    std::vector<int> flag;
    // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
    flag.resize(GetCurrentNodeGeneId() / sizeof(int) + 1, 0);
    std::stack<NodeGeneId> stack;
    stack.push(srcNode);
    while (!stack.empty())
    {
        const NodeGeneId n = stack.top();
        stack.pop();

        if (genome.incomingConnectionList.find(n) == genome.incomingConnectionList.end())
        {
            continue;
        }

        for (const InnovationId innovId : genome.incomingConnectionList.at(n))
        {
            const ConnectionGene* con = GetConnectionGene(genome, innovId);
            assert(con);

            const NodeGeneId inNode = con->inNode;

            if (inNode == targetNode)
            {
                return false;
            }

            const int index = inNode / sizeof(int);
            const int offset = inNode % sizeof(int);
            if (((flag[index] >> offset) & 1) == 0)
            {
                stack.push(inNode);
                flag[index] |= 1 << offset;
            }
        }
    }

    return true;
}

bool NEAT::SameConnectionExist(const Genome& genome, NodeGeneId inNode, NodeGeneId outNode) const
{
    for (const auto& con : genome.connectionGenes)
    {
        if (con.inNode == inNode && con.outNode == outNode)
        {
            return true;
        }
    }
    return false;
}

// Perform cross over operation over two genomes and generate a new genome
// This function assumes that genome1 has a higher fitness value than genome2
// Set sameFitness true when genome1 and genome2 have the same fitness values
auto NEAT::CrossOver(const Genome& genome1, float fitness1, const Genome& genome2, float fitness2) const -> Genome
{
    const Genome* parent1 = &genome1;
    const Genome* parent2 = &genome2;
    const bool sameFitness = fitness1 == fitness2;
    if (fitness1 < fitness2)
    {
        parent1 = &genome2;
        parent2 = &genome1;
    }

    // Create a new genome
    Genome child;

    auto TryAddConnection = [this, &child](const ConnectionGene& connection, bool enable)
    {
        if (!SameConnectionExist(child, connection.inNode, connection.outNode) &&
            (config.allowCyclicNetwork || CheckCyclic(child, connection.inNode, connection.outNode)))
        {
            child.connectionGenes.push_back(connection);
            child.connectionGenes.back().enabled = enable;
            child.incomingConnectionList[connection.outNode].push_back(connection.innovId);
            child.outgoingConnectionList[connection.inNode].push_back(connection.innovId);
        }
    };

    // Inherite connection genes from the two parents
    {
        RandomIntDistribution<int> random(0, 1);
        RandomRealDistribution<float> randomf(0, 1.f);

        const auto& cGenes1 = parent1->connectionGenes;
        const auto& cGenes2 = parent2->connectionGenes;
        const size_t numConnectionsP1 = cGenes1.size();
        const size_t numConnectionsP2 = cGenes2.size();

        child.connectionGenes.reserve(std::min(numConnectionsP1, numConnectionsP2));

        size_t ip1 = 0, ip2 = 0;

        // Select each gene from either parent based on gene's innovation id and parents' fitness
        // Note that cGenes1/2 are sorted by innovation id
        while (ip1 < numConnectionsP1 && ip2 < numConnectionsP2)
        {
            const ConnectionGene& cGene1 = cGenes1[ip1];
            const ConnectionGene& cGene2 = cGenes2[ip2];
            const ConnectionGene* geneToInherite = nullptr;

            bool enabled = true;

            // When two genes have the same innovation id, take one gene randomly from either parent
            // Or if two genes have the same fittness values, always take one gene randomly from either parent regardless of innovation id
            bool sameInnovation = cGene1.innovId == cGene2.innovId;
            if (sameInnovation)
            {
                geneToInherite = random(randomGenerator) ? &cGene1 : &cGene2;

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
                // If this gene exists only in parent1, inherite from parent1
                if (cGene1.innovId < cGene2.innovId)
                {
                    geneToInherite = !sameFitness || random(randomGenerator) ? &cGene1 : nullptr;
                    ++ip1;
                }
                // If this gene existss only in parent2, don't inherite it
                else
                {
                    geneToInherite = sameFitness && random(randomGenerator) ? &cGene2 : nullptr;
                    ++ip2;
                }
            }

            if (geneToInherite)
            {
                // Add connection gene to the new child
                TryAddConnection(*geneToInherite, enabled);
            }
        }

        // Add remaining genes
        {
            auto addRemainingGenes = [this, sameFitness, &child, TryAddConnection](size_t index, size_t num, const ConnectionGeneList& genes, bool inherite)
            {
                while (index < num)
                {
                    if (inherite)
                    {
                        TryAddConnection(genes[index], genes[index].enabled);
                    }
                    ++index;
                }
            };

            if (ip1 < numConnectionsP1)
            {
                if (!sameFitness || random(randomGenerator))
                {
                    addRemainingGenes(ip1, numConnectionsP1, cGenes1, true);
                }
            }
            else if (ip2 < numConnectionsP2)
            {
                if (sameFitness && random(randomGenerator))
                {
                    addRemainingGenes(ip2, numConnectionsP2, cGenes2, sameFitness);
                }
            }
        }
    }

    // Add node genes to the child genome
    {
        // Create a map of NodeGenes and their ids first
        std::unordered_map<NodeGeneId, NodeGene> nodeGenes;
        {
            auto addNodeGeneIfNotExist = [&nodeGenes, &parent1, &parent2, this](NodeGeneId id)
            {
                if (nodeGenes.find(id) == nodeGenes.end())
                {
                    NodeGene newNode;
                    newNode.id = id;
                    const NodeGene* pn = GetNodeGene(*parent1, id);
                    if (!pn)
                    {
                        pn = GetNodeGene(*parent2, id);
                    }
                    newNode.type = pn->type;
                    newNode.activationFuncId = pn->activationFuncId;
                    nodeGenes[id] = newNode;
                }
            };

            for (const auto& connectionGene : child.connectionGenes)
            {
                addNodeGeneIfNotExist(connectionGene.inNode);
                addNodeGeneIfNotExist(connectionGene.outNode);
            }
        }

        // Add all the node genes to the child genome
        child.nodeGenes.reserve(nodeGenes.size());
        for (const auto& elem : nodeGenes)
        {
            child.nodeGenes.push_back(elem.second);
        }

        std::sort(child.nodeGenes.begin(), child.nodeGenes.end(),
            [](const NodeGene& a, const NodeGene& b)
            {
                return a.id < b.id;
            });
    }

#ifdef _DEBUG
    CheckSanity(child);
#endif

    return child;
}

// Implementation of generating a new generation
void NEAT::GenerateNewGeneration()
{
    Mutate(generation);

    // Copy the current generation
    Generation curGen = generation;

    // Evaluate genomes and copy high score organizations to the next gen
    struct Score
    {
        float fitness;
        uint32_t index;
        bool operator< (const Score& rhs) { return fitness > rhs.fitness; }
    };

    // Evaluate all the current gen genomes
    std::vector<Score> scores;
    scores.reserve(config.numOrganismsInGeneration);
    for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        scores.push_back(Score{ Evaluate(curGen.genomes[i]), i });
    }

    if (config.diversityProtection == DiversityProtectionMethod::Speciation)
    {
        std::vector<int> divs;
        divs.resize(config.numOrganismsInGeneration);
        for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
        {
            for (uint32_t j = i; j < config.numOrganismsInGeneration; ++j)
            {
                if (i == j)
                {
                    divs[i]++;
                    continue;
                }

                if (CalculateDistance(curGen.genomes[i], curGen.genomes[j]) < config.speciationDistThreshold)
                {
                    divs[i]++;
                    divs[j]++;
                }
            }

            scores[i].fitness /= (float)divs[i];
        }
    }
    else if (config.diversityProtection == DiversityProtectionMethod::MorphologicalInnovationProtection)
    {
        // Not implemented yet
        assert(0);
    }

    // Sort genomes by fitness
    std::sort(scores.begin(), scores.end());

    // Just copy high score genomes to the next generation
    const int numOrgsToCopy = int(config.numOrganismsInGeneration * (1.f - config.crossOverRate));
    for (int i = 0; i < numOrgsToCopy; ++i)
    {
        generation.genomes[i] = curGen.genomes[scores[i].index];
    }

    RandomIntDistribution<int> randomi(0, config.numOrganismsInGeneration - 1);
    if (config.enableCrossOver)
    {
        // Rest population will be generated by cross over
        for (uint32_t i = numOrgsToCopy; i < config.numOrganismsInGeneration; ++i)
        {
            // Select random two genomes
            int i1 = randomi(randomGenerator);
            int i2 = randomi(randomGenerator);
            while (i1 == i2)
            {
                i2 = randomi(randomGenerator);
            }

            // Ensure the genome at i1 has a higher fitness
            if (scores[i1] < scores[i2])
            {
                std::swap(i1, i2);
            }

            // Cross over
            generation.genomes[i] = CrossOver(
                curGen.genomes[scores[i1].index],
                scores[i1].fitness,
                curGen.genomes[scores[i2].index],
                scores[i2].fitness);
        }
    }
    else
    {
        // Just randomly select the rest population
        for (uint32_t i = numOrgsToCopy; i < config.numOrganismsInGeneration; ++i)
        {
            generation.genomes[i] = curGen.genomes[randomi(randomGenerator)];
        }
    }

    generation.generationId++;
}

void NEAT::Mutate(Generation& g) const
{
    RandomRealDistribution<float> randomf(-1.f, 1.f);
    NewlyAddedNodes newNodes;
    NewlyAddedConnections newConnections;

    // Mutation weights
    for (auto& genome : g.genomes)
    {
        for (auto& connection : genome.connectionGenes)
        {
            // Invoke mutation at random rate
            if (randomf(randomGenerator) <= config.weightMutationRate)
            {
                // Only perturbate weight at random rate
                if (randomf(randomGenerator) <= config.weightPerturbationRate)
                {
                    connection.weight = randomf(randomGenerator) * 0.025f;
                }
                // Otherwise assign a completely new weight
                else
                {
                    connection.weight = randomf(randomGenerator);
                }
            }
        }
    }

    // Apply topological mutations
    for (auto& genome : g.genomes)
    {
        // Add a new node at random rate
        if (randomf(randomGenerator) <= config.nodeAdditionRate)
        {
            NewlyAddedNode node = AddNewNode(genome);
            if (node.node != invalidGeneId)
            {
                newNodes[node.innovId].push_back(node);
            }
        }

        // Add a new connection at random rate
        if (randomf(randomGenerator) <= config.connectionAdditionRate)
        {
            NewlyAddedConnection con = AddNewConnection(genome);
            if (con.innovId != invalidInnovationId)
            {
                newConnections[con.nodes].push_back(con);
            }
        }
    }

    EnsureUniqueGeneIndices(newNodes, newConnections);
}

// Make sure that the same topological changes have the same id
void NEAT::EnsureUniqueGeneIndices(const NewlyAddedNodes& newNodes, const NewlyAddedConnections& newConnections) const
{
    // Check duplicated newly added node genes
    for (auto& elem : newNodes)
    {
        auto& genomes = elem.second;
        NodeGeneId id = genomes[0].node;
        for (size_t i = 1; i < genomes.size(); ++i)
        {
            NodeGeneId thisId = genomes[i].node;
            Genome& genome = genomes[i].genome;

            // Here, we are assuming that only one new node can be added by mutation at a genome in a single generation
            // Both node genes are just added in this generation, so their id should be greater than any of existing node gene.
            // So node gene should be still sorted even after this operation.
            assert(genome.nodeGenes[genome.nodeGenes.size() - 1].id == thisId);
            genome.nodeGenes[genome.nodeGenes.size() - 1].id = id;

            // Again here, we are assuming that only at most three connection genes can be added by mutation at a genome in a single generation
            // So it's enough to check the last three connection genes to update
            int firstIndex = genome.connectionGenes.size() - 3;
            if (firstIndex < 0) firstIndex = 0;
            for (size_t i = firstIndex; i < genome.connectionGenes.size(); ++i)
            {
                auto& connection = genome.connectionGenes[i];
                if (connection.inNode == thisId)
                {
                    connection.inNode = id;
                }
                if (connection.outNode == thisId)
                {
                    connection.outNode = id;
                }
            }

            // Reassign node connection map to the new node gene id
            assert(genome.incomingConnectionList.find(thisId) != genome.incomingConnectionList.end());
            assert(genome.incomingConnectionList.find(id) == genome.incomingConnectionList.end());
            assert(genome.outgoingConnectionList.find(thisId) != genome.outgoingConnectionList.end());
            assert(genome.outgoingConnectionList.find(id) == genome.outgoingConnectionList.end());
            genome.incomingConnectionList[id] = genome.incomingConnectionList[thisId];
            genome.incomingConnectionList.erase(thisId);
            genome.outgoingConnectionList[id] = genome.outgoingConnectionList[thisId];
            genome.outgoingConnectionList.erase(thisId);

#ifdef _DEBUG
            CheckSanity(genome);
#endif
        }
    }

    // Check duplicated newly added connection ids
    for (auto& elem : newConnections)
    {
        auto& genomes = elem.second;
        InnovationId id = genomes[0].innovId;
        for (size_t i = 1; i < genomes.size(); ++i)
        {
            InnovationId thisId = genomes[i].innovId;
            Genome& genome = genomes[i].genome;

            {
                // Again here, we are assuming that only at most three connection genes can be added by mutation at a genome in a single generation
                // So it's enough to check the last three connection genes to update
                int firstIndex = genome.connectionGenes.size() - 3;
                if (firstIndex < 0) firstIndex = 0;
                for (size_t i = firstIndex; i < genome.connectionGenes.size(); ++i)
                {
                    auto& connection = genome.connectionGenes[i];
                    if (connection.innovId == thisId)
                    {
                        connection.innovId = id;
                    }
                }

                // This time, innovation ids for the last three connection can be not sorted anymore.
                // So we sort them.
                for (size_t i = firstIndex + 1; i < genome.connectionGenes.size(); ++i)
                {
                    size_t j = i;
                    while (genome.connectionGenes[j - 1].innovId > genome.connectionGenes[j].innovId)
                    {
                        auto tmp = genome.connectionGenes[j - 1];
                        genome.connectionGenes[j - 1] = genome.connectionGenes[j];
                        genome.connectionGenes[j] = tmp;
                        --j;
                    }
                }
            }

            // Update node connection map
            auto updateConnectionIdBruteForce = [thisId, id](NodeConnectionMap& map)
            {
                for (auto& ids : map)
                {
                    for (size_t j = 0; j < ids.second.size(); ++j)
                    {
                        if (ids.second[j] == thisId)
                        {
                            ids.second[j] = id;
                        }
                    }
                }
            };
            updateConnectionIdBruteForce(genome.incomingConnectionList);
            updateConnectionIdBruteForce(genome.outgoingConnectionList);

#ifdef _DEBUG
            CheckSanity(genome);
#endif
        }
    }
}

// Calculate distance between two genomes based on their topologies and weights
float NEAT::CalculateDistance(const Genome& genome1, const Genome& genome2) const
{
    const auto& cGenes1 = genome1.connectionGenes;
    const auto& cGenes2 = genome2.connectionGenes;
    const size_t numConnectionsP1 = cGenes1.size();
    const size_t numConnectionsP2 = cGenes2.size();
    size_t ip1, ip2;

    const size_t numConnections = numConnectionsP1 > numConnectionsP2 ? numConnectionsP1 : numConnectionsP2;
    size_t numMismatches = 0;
    float weightDifference = 0.f;

    for (ip1 = 0, ip2 = 0; ip1 < numConnectionsP1 && ip2 < numConnectionsP2;)
    {
        const ConnectionGene& cGene1 = cGenes1[ip1];
        const ConnectionGene& cGene2 = cGenes2[ip2];
        if (cGene1.innovId == cGene2.innovId)
        {
            weightDifference += std::fabs(cGene1.weight - cGene2.weight);
            ++ip1;
            ++ip2;
        }
        else
        {
            numMismatches++;
            if (cGene1.innovId < cGene2.innovId)
            {
                ++ip1;
            }
            else
            {
                ++ip2;
            }
        }
    }

    return (float)numMismatches / (float)numConnections + 0.4f * weightDifference;
}

// Evaluate a genome and return its fitness
float NEAT::Evaluate(const Genome& genom) const
{
    if (fitnessFunc)
    {
        return fitnessFunc(genom);
    }

    return 0.0f;
}

// Evaluate value of each node recursively
void NEAT::EvaluateRecursive(const Genome& genome, NodeGeneId nodeId, std::vector<NodeGeneId>& evaluatingNodes, std::unordered_map<NodeGeneId, float>& values) const
{
    assert(config.allowCyclicNetwork == false);

    float val = 0.f;

    if (genome.incomingConnectionList.find(nodeId) != genome.incomingConnectionList.end())
    {
        const auto& incomings = genome.incomingConnectionList.at(nodeId);
        for (auto innovId : incomings)
        {
            const auto connectionGene = GetConnectionGene(genome, innovId);
            assert(connectionGene != nullptr);

            if (!connectionGene->enabled) continue;

            auto incomingNodeId = connectionGene->inNode;
            bool alreadyEvaluating = false;

            if (config.allowCyclicNetwork)
            {
                for (NodeGeneId id : evaluatingNodes)
                {
                    if (incomingNodeId == id)
                    {
                        alreadyEvaluating = true;
                        break;
                    }
                }
                if (alreadyEvaluating) continue;
                evaluatingNodes.push_back(incomingNodeId);
            }

            if (values.find(incomingNodeId) == values.end())
            {
                EvaluateRecursive(genome, incomingNodeId, evaluatingNodes, values);
            }

            if (config.allowCyclicNetwork)
            {
                evaluatingNodes.resize(evaluatingNodes.size() - 1);
            }

            val += values[incomingNodeId] * connectionGene->weight;
        }
    }

    ActivationFuncId activation;
    if (config.useGlobalActivationFunc)
    {
        activation = defaultActivationFuncId;
    }
    else
    {
        activation = GetNodeGene(genome, nodeId)->activationFuncId;
    }

    values[nodeId] = activationFuncs[activation](val);
}

// Create default genome for the initial generation
auto NEAT::CreateDefaultInitialGenome() const -> Genome
{
    Genome genome;

    genome.nodeGenes.reserve(3);
    auto& nodeGenes = genome.nodeGenes;

    // Input node
    nodeGenes.push_back(NodeGene{ 0, NodeGeneType::Input, defaultActivationFuncId });

    // Hidden node
    nodeGenes.push_back(NodeGene{ 1, NodeGeneType::Hidden, defaultActivationFuncId });

    // Output node
    nodeGenes.push_back(NodeGene{ 2, NodeGeneType::Output, defaultActivationFuncId });

    genome.connectionGenes.reserve(2);
    auto& connections = genome.connectionGenes;

    // Create a connection between the input and the hidden node
    // InnovationId of this connection for default init genom will be always 0
    RandomRealDistribution<float> randomf(-1.f, 1.f);
    connections.push_back(ConnectionGene{ 0, 0, 1, randomf(randomGenerator), true });
    genome.outgoingConnectionList[0].push_back(0);
    genome.incomingConnectionList[1].push_back(0);

    // Create a connection between the hidden and the output node
    // InnovationId of this connection for default init genom will be always 1
    connections.emplace_back(ConnectionGene{ 1, 1, 2, randomf(randomGenerator), true });
    genome.outgoingConnectionList[1].push_back(1);
    genome.incomingConnectionList[2].push_back(1);

    return genome;
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

auto NEAT::GetConnectionGene(Genome& genome, InnovationId innovId) const -> ConnectionGene*
{
    int index = FindGeneBinarySearch(
        genome.connectionGenes,
        innovId,
        [](const ConnectionGene& gene){ return gene.innovId; });
    if (index < 0) return nullptr;
    return &genome.connectionGenes[index];
}

auto NEAT::GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*
{
    int index = FindGeneBinarySearch(
        genome.connectionGenes,
        innovId,
        [](const ConnectionGene& gene){ return gene.innovId; });
    if (index < 0) return nullptr;
    return &genome.connectionGenes[index];
}

auto NEAT::GetNodeGene(const Genome& genome, NodeGeneId id) const -> const NodeGene*
{
    int index = FindGeneBinarySearch(
        genome.nodeGenes,
        id,
        [](const NodeGene & gene){ return gene.id; });
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

#ifdef _DEBUG
bool NEAT::CheckSanity(const Genome& genome) const
{
    if (!config.enableSanityCheck)
    {
        return true;
    }

    // Check if node gene ids are sorted
    {
        const auto& nodes = genome.nodeGenes;
        const size_t numOfNode = nodes.size();
        for (size_t i = 1; i < numOfNode; ++i)
        {
            if (nodes[i - 1].id >= nodes[i].id)
            {
                return false;
            }
        }
    }

    // Check if innovation ids are sorted
    {
        const auto& conns = genome.connectionGenes;
        const size_t numOfConns = conns.size();
        for (size_t i = 1; i < numOfConns; ++i)
        {
            if (conns[i - 1].innovId >= conns[i].innovId)
            {
                return false;
            }
        }
    }

    // Check if there's no duplicated connection genes for the same in and out nodes
    {
        const auto& conns = genome.connectionGenes;
        const size_t numOfConns = conns.size();
        for (size_t i = 0; i < numOfConns; ++i)
        {
            for (size_t j = i + 1; j < numOfConns; ++j)
            {
                if (conns[i].inNode == conns[j].inNode && conns[i].outNode == conns[j].outNode)
                {
                    return false;
                }
            }
        }
    }

    // Check if node-connection map is consistent
    {
        const auto& incomings = genome.incomingConnectionList;
        for (const auto& elem : incomings)
        {
            for (auto innovId : elem.second)
            {
                auto con = GetConnectionGene(genome, innovId);
                if (!con || con->outNode != elem.first)
                {
                    return false;
                }
            }
        }
    }

    // Check if node-connection map is consistent
    {
        const auto& outgoings = genome.outgoingConnectionList;
        for (const auto& elem : outgoings)
        {
            for (auto innovId : elem.second)
            {
                auto con = GetConnectionGene(genome, innovId);
                if (!con || con->inNode != elem.first)
                {
                    return false;
                }
            }
        }
    }

    // Bias node shouldn't have incoming connections
    {
        const auto& nodes = genome.nodeGenes;
        const size_t numOfNode = nodes.size();
        for (size_t i = 0; i < numOfNode; ++i)
        {
            if (nodes[i].type == NodeGeneType::Bias)
            {
                if (genome.incomingConnectionList.find(nodes[i].id) != genome.incomingConnectionList.end())
                {
                    return false;
                }
            }
        }
    }

    if (!config.allowCyclicNetwork)
    {
        for (const auto& con : genome.connectionGenes)
        {
            if (!CheckCyclic(genome, con.inNode, con.outNode))
            {
                return false;
            }
        }
    }

    return true;
}

#endif

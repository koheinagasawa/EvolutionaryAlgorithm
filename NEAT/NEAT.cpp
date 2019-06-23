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

    SetupInitialNodeGenes();

    // Populate the first generation with default genoms
    generation.genomes = std::make_shared<std::vector<Genome>>();
    generation.genomes->resize(config.numOrganismsInGeneration);
    genomeListBuffer = std::make_shared<std::vector<Genome>>();
    genomeListBuffer->resize(config.numOrganismsInGeneration);
    for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        (*generation.genomes)[i] = CreateDefaultInitialGenome();
    }

    {
        const Genome& g = (*generation.genomes)[0];
        currentInnovationId = g.connectionGenes.size();

        for (const auto& elem : g.connectionGenes)
        {
            innovationHistory[NodePair{ elem.second.inNode, elem.second.outNode }] = elem.second;
        }
    }

    if (config.activateFunctions.size() == 0 || config.activateFunctions.size() <= config.defaultActivateFunctionId)
    {
        // TODO: Add some useful warning message here
        return generation;
    }

    activationFuncs = config.activateFunctions;
    defaultActivationFuncId = config.defaultActivateFunctionId;

    scores.resize(config.numOrganismsInGeneration);

    isInitialized = true;

    return generation;
}

// Reset all the state in NEAT
// After calling this function, Initialize has to be called in order to run NEAT again
void NEAT::Reset()
{
    generation = Generation();
    currentInnovationId = 0;
    currentSpeciesId = 0;
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
    const int size = generation.genomes->size();
    float maxFitness = 0;
    int bestGenomeId = -1;
    int speciesOfTheBestGenome = -1;
    float sumScoreOfBestSpecies = 0.f;
    for (int i = 0; i < size; ++i)
    {
        const Genome& genome = (*generation.genomes).at(i);
        const float fitness = Evaluate(genome);
        //std::cout << "Genome " << i << ": " << fitness << std::endl;
        sum += fitness;
        if (fitness > maxFitness)
        {
            maxFitness = fitness;
            bestGenomeId = i;
            speciesOfTheBestGenome = genome.species;
        }

        if (genome.species == generation.species[0].id)
        {
            sumScoreOfBestSpecies += fitness;
        }
    }
    std::cout << "Average fitness over " << size << " organims: " << sum / (float)size << std::endl;
    std::cout << "Maximum fitness: Species " << speciesOfTheBestGenome << ": Genome " << bestGenomeId << " - " << maxFitness << std::endl;
    std::cout << "Number of species: " << generation.species.size() << std::endl;
    assert(speciesOfTheBestGenome == generation.species[0].id);
    std::cout << "Number of genomes in the best species: " << generation.species[0].genomes.size() << std::endl;
    std::cout << "Averaged score of the best species: " << sumScoreOfBestSpecies / (float)generation.species[0].genomes.size() << std::endl;
}

auto NEAT::GetNewInnovationId() -> InnovationId
{
    // TODO: Make it thread safe
    return currentInnovationId++;
}

auto NEAT::CreateNewNode(NodeGeneType type) -> NodeGeneId
{
    NodeGene node{ type };

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

    NodeGeneId nodeGeneId = generation.nodeGenes.size();
    generation.nodeGenes.push_back(node);

    return nodeGeneId;
}

// Add a new node at a random connection
auto NEAT::AddNewNode(Genome& genome) -> NodeAddedInfo
{
    // Collect all connections where we can add new node first
    std::vector<InnovationId> availableConnections;
    {
        availableConnections.reserve(genome.connectionGenes.size());

        for (const auto& elem : genome.connectionGenes)
        {
            const ConnectionGene& gene = elem.second;
            if (gene.enabled)
            {
                availableConnections.push_back(gene.innovId);
            }
        }

        // Return when there's no available connection to add a new node
        if (availableConnections.size() == 0)
        {
            // TODO: Output a useful message
            return NodeAddedInfo{genome, invalidNodeGeneId};
        }
    }

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

    // Create a new node
    NodeGeneId newNodeId = CreateNewNode(NodeGeneType::Hidden);
    genome.nodeLinks.insert({ newNodeId, Genome::Links() });

    // Create a new connection between the inNode and the new node
    InnovationId newCon1 = Connect(genome, inNode, newNodeId, 1.f);

    // Create a new connection between the new node and the outNode
    InnovationId newCon2 = Connect(genome, newNodeId, outNode, weight);

    assert(CheckSanity(genome));

    return NodeAddedInfo{ genome, newNodeId, innovId, newCon1, newCon2 };
}

auto NEAT::Connect(Genome& genome, NodeGeneId inNode, NodeGeneId outNode, float weight) -> InnovationId
{
    assert(genome.nodeLinks.find(inNode) != genome.nodeLinks.end());
    assert(genome.nodeLinks.find(outNode) != genome.nodeLinks.end());

    ConnectionGene gene;
    gene.inNode = inNode;
    gene.outNode = outNode;

    NodePair pair(inNode, outNode);
    if (innovationHistory.find(pair) == innovationHistory.end())
    {
        gene.innovId = GetNewInnovationId();
        innovationHistory[pair] = gene;
    }
    else
    {
        gene.innovId = innovationHistory.at(pair).innovId;
    }

    gene.enabled = true;
    gene.weight = weight;
    assert(genome.connectionGenes.find(gene.innovId) == genome.connectionGenes.end());
    genome.connectionGenes[gene.innovId] = gene;

    genome.nodeLinks[inNode].outgoings.push_back(gene.innovId);
    genome.nodeLinks[outNode].incomings.push_back(gene.innovId);

    return gene.innovId;
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
    // Select out node first
    NodeGeneId outNodeId = invalidNodeGeneId;
    {
        // Collect all node genes where we can add a new connection gene first
        std::vector<NodeGeneId> nodeCandidates = GetNodeCandidatesAsOutNodeOfNewConnection(genome);

        // Return when there's no available node to add a new connection
        if (nodeCandidates.size() > 0)
        {
            // Select a random node from the available ones as inNode
            outNodeId = SelectRandomNodeGene(nodeCandidates);
        }
    }

    if (outNodeId == invalidNodeGeneId)
    {
        // TODO: Output a useful message
        return;
    }

    NodeGeneId inNodeId = invalidNodeGeneId;
    {
        std::vector<NodeGeneId> nodeCandidates;
        // Then collect all node genes which are not connected to the outNode
        {
            std::vector<int> connectedFlag;
            // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
            const int numAllNodes = (int)generation.nodeGenes.size();
            connectedFlag.resize((numAllNodes / sizeof(int)) + 1, 0);

            for (auto innovId : genome.nodeLinks.at(outNodeId).incomings)
            {
                const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
                assert(cgene);
                const NodeGeneId nodeId = cgene->inNode;
                connectedFlag[nodeId / sizeof(int)] |= (1 << (nodeId % sizeof(int)));
            }

            // Find all nodes not connected to node1
            for (const auto& elem : genome.nodeLinks)
            {
                const NodeGeneId node = elem.first;
                if (node == outNodeId ||
                    generation.nodeGenes[node].type == NodeGeneType::Output)
                {
                    continue;
                }

                if (((connectedFlag[node / sizeof(int)] >> (node % sizeof(int))) & 1) == 0)
                {
                    nodeCandidates.push_back(node);
                }
            }
        }

        // availableNodes must not empty here because node1 has at least one other node which it's not connected to
        assert(nodeCandidates.size() > 0);

        // Select a random node from the available ones as outNode
        inNodeId = SelectRandomNodeGene(nodeCandidates);
    }

    // Add a new connection
    RandomRealDistribution<float> floatDistribution(-1.f, 1.f);
    auto innovId = Connect(genome, inNodeId, outNodeId, floatDistribution(randomGenerator));

    assert(CheckSanity(genome));
}

// Add a new connection between random two nodes without allowing cyclic network
// Direction of the new connection is guaranteed to be one direction (distance from in node to an input node is smaller than the one of out node)
void NEAT::AddNewForwardConnection(Genome& genome)
{
    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> outNodeCandidates = GetNodeCandidatesAsOutNodeOfNewConnection(genome);

        // Return when there's no available node to add a new connection
    if (outNodeCandidates.size() == 0)
    {
        // TODO: Output a useful message message
        return;
    }

    // Randomize the order of available nodes in the array
    std::random_shuffle(outNodeCandidates.begin(), outNodeCandidates.end());

    NodeGeneId inNodeId = invalidNodeGeneId;
    NodeGeneId outNodeId = invalidNodeGeneId;
    for (auto onid : outNodeCandidates)
    {
        std::vector<NodeGeneId> inNodeCandidates;

        std::vector<int> connectedFlag;
        // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
        const int numAllNodes = (int)generation.nodeGenes.size();
        connectedFlag.resize(numAllNodes / sizeof(int) + 1, 0);

        for (auto innovId : genome.nodeLinks[onid].incomings)
        {
            const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
            assert(cgene);
            const NodeGeneId nodeId = cgene->inNode;
            connectedFlag[nodeId / sizeof(int)] |= (1 << (nodeId % sizeof(int)));
        }

        for (const auto& elem : genome.nodeLinks)
        {
            const NodeGeneId node = elem.first;
            if (node == outNodeId ||
                generation.nodeGenes[node].type == NodeGeneType::Output)
            {
                continue;
            }

            if (((connectedFlag[node / sizeof(int)] >> (node % sizeof(int))) & 1) == 0)
            {
                if (CheckCyclic(genome, node, onid))
                {
                    inNodeCandidates.push_back(node);
                }
            }
        }

        if (inNodeCandidates.size() == 0)
        {
            // No available node where we can create a forward connection found
            continue;
        }

        outNodeId = onid;
        // Select a random node from the available ones as outNode
        inNodeId = SelectRandomNodeGene(inNodeCandidates);

        break;
    }

    // Return when there's no available nodes
    if (outNodeId == invalidNodeGeneId)
    {
        // TODO: Output a useful message
        return;
    }

    // Add a new connection
    RandomRealDistribution<float> floatDistribution(-1.f, 1.f);
    auto innovId = Connect(genome, inNodeId, outNodeId, floatDistribution(randomGenerator));

    assert(CheckSanity(genome));
}

auto NEAT::GetNodeCandidatesAsOutNodeOfNewConnection(const Genome& genome) const -> std::vector<NodeGeneId>
{
    // Collect all node genes where we can add a new connection gene
    std::vector<NodeGeneId> out;
    const size_t numNodes = genome.nodeLinks.size();
    for (const auto& elem : genome.nodeLinks)
    {
        const NodeGeneId node = elem.first;
        if (generation.nodeGenes[node].type != NodeGeneType::Input &&
            generation.nodeGenes[node].type != NodeGeneType::Bias &&
            genome.nodeLinks.at(node).incomings.size() < numNodes)
        {
            out.push_back(node);
        }
    }
    return out;
}

bool NEAT::CheckCyclic(const Genome& genome, NodeGeneId srcNode, NodeGeneId targetNode) const
{
    if (srcNode == targetNode)
    {
        return false;
    }

    std::vector<int> flag;
    // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
    const int numAllNodes = generation.nodeGenes.size();
    flag.resize(numAllNodes / sizeof(int) + 1, 0);
    std::stack<NodeGeneId> stack;
    stack.push(srcNode);
    while (!stack.empty())
    {
        const NodeGeneId node = stack.top();
        stack.pop();

        for (const InnovationId innovId : genome.nodeLinks.at(node).incomings)
        {
            const ConnectionGene* con = GetConnectionGene(genome, innovId);
            assert(con);

            if (!con->enabled)
            {
                continue;
            }

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
        bool newInNode = child.nodeLinks.find(connection.inNode) == child.nodeLinks.end();
        bool newOutNode = child.nodeLinks.find(connection.outNode) == child.nodeLinks.end();
        if (newInNode)
        {
            child.nodeLinks.insert({ connection.inNode, Genome::Links() });
        }
        if (newOutNode)
        {
            child.nodeLinks.insert({ connection.outNode, Genome::Links() });
        }

        if (config.allowCyclicNetwork || CheckCyclic(child, connection.inNode, connection.outNode))
        {
            ConnectionGene newCon = connection;
            newCon.enabled = enable;
            child.connectionGenes[newCon.innovId] = newCon;
            child.nodeLinks[connection.outNode].incomings.push_back(connection.innovId);
            child.nodeLinks[connection.inNode].outgoings.push_back(connection.innovId);
        }
        else
        {
            if (newInNode)
            {
                child.nodeLinks.erase(connection.inNode);
            }
            if (newOutNode)
            {
                child.nodeLinks.erase(connection.outNode);
            }
        }
    };

    // Inherite connection genes from the two parents
    {
        RandomIntDistribution<int> random(0, 1);
        RandomRealDistribution<float> randomf(0, 1.f);

        const auto& cGenes1 = parent1->connectionGenes;
        const auto& cGenes2 = parent2->connectionGenes;

        auto itr1 = cGenes1.begin();
        auto itr2 = cGenes2.begin();

        // Select each gene from either parent based on gene's innovation id and parents' fitness
        // Note that cGenes1/2 are sorted by innovation id
        while (itr1 != cGenes1.end() && itr2 != cGenes2.end())
        {
            const ConnectionGene& cGene1 = itr1->second;
            const ConnectionGene& cGene2 = itr2->second;
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

                ++itr1;
                ++itr2;
            }
            else
            {
                // If this gene exists only in parent1, inherite from parent1
                if (cGene1.innovId < cGene2.innovId)
                {
                    geneToInherite = !sameFitness || random(randomGenerator) ? &cGene1 : nullptr;
                    ++itr1;
                }
                // If this gene existss only in parent2, don't inherite it
                else
                {
                    geneToInherite = sameFitness && random(randomGenerator) ? &cGene2 : nullptr;
                    ++itr2;
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
            auto addRemainingGenes = [this, sameFitness, &child, &random, TryAddConnection](ConnectionGeneList::iterator itr, const ConnectionGeneList& genes)
            {
                while (itr != genes.end())
                {
                    if (sameFitness && random(randomGenerator))
                    {
                        TryAddConnection(itr->second, itr->second.enabled);
                    }
                    ++itr;
                }
            };

            while (itr1 != cGenes1.end())
            {
                if (!sameFitness && random(randomGenerator))
                {
                    TryAddConnection(itr1->second, itr1->second.enabled);
                }
                ++itr1;
            }
            while (itr2 != cGenes2.end())
            {
                if (sameFitness && random(randomGenerator))
                {
                    TryAddConnection(itr2->second, itr2->second.enabled);
                }
                ++itr2;
            }
        }
    }

    child.species = parent1->species;

    assert(CheckSanity(child));

    return child;
}

// Implementation of generating a new generation
void NEAT::GenerateNewGeneration()
{
    Mutate(generation);

    GenomeList currentGen = generation.genomes;
    GenomeList nextGen = genomeListBuffer;

    for (uint32_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        scores[i] = Score{ Evaluate((*currentGen)[i]), i };
    }

    std::vector<int> genomesToCopy;
    size_t numOrgsToCopy = size_t(config.numOrganismsInGeneration * (1.f - config.crossOverRate));

    if (config.diversityProtection == DiversityProtectionMethod::Speciation)
    {
        Speciation(generation, scores, genomesToCopy);
    }
    else if (config.diversityProtection == DiversityProtectionMethod::MorphologicalInnovationProtection)
    {
        // Not implemented yet
        assert(0);
    }

    size_t nextGenomeIndex = 0;
    for (; nextGenomeIndex < genomesToCopy.size(); ++nextGenomeIndex)
    {
        (*nextGen)[nextGenomeIndex] = (*currentGen)[genomesToCopy[nextGenomeIndex]];
    }

    // Sort genomes by fitness
    std::sort(scores.begin(), scores.end(), [](const Score& s1, const Score& s2)
        {
            return s1.fitness > s2.fitness;
        });

    {
        float scoreSum = 0;
        for (const auto& score : scores)
        {
            scoreSum += score.fitness;
        }
        auto getGenome = [this, scoreSum, currentGen](float f)
        {
            float currentSum = 0;
            for (const auto& score : scores)
            {
                currentSum += score.fitness;
                if (currentSum > f)
                {
                    return (*currentGen)[score.index];
                }
            }

            return (*currentGen)[scores.back().index];
        };

        // Just copy high score genomes to the next generation
        RandomRealDistribution<float> random(0, scoreSum);
        for (; nextGenomeIndex < numOrgsToCopy; ++nextGenomeIndex)
        {
            (*nextGen)[nextGenomeIndex] = getGenome(random(randomGenerator));
        }
    }

    RandomIntDistribution<int> randomi(0, config.numOrganismsInGeneration - 1);
    if (config.enableCrossOver)
    {
        // Rest population will be generated by cross over
        for (; nextGenomeIndex < config.numOrganismsInGeneration; ++nextGenomeIndex)
        {
            // Select random two genomes
            int i1 = randomi(randomGenerator);
            int i2 = randomi(randomGenerator);
            while (i1 == i2)
            {
                i2 = randomi(randomGenerator);
            }

            // Ensure the genome at i1 has a higher fitness
            if (scores[i1].fitness < scores[i2].fitness)
            {
                std::swap(i1, i2);
            }

            // Cross over
            (*nextGen)[nextGenomeIndex] = CrossOver(
                (*currentGen)[scores[i1].index], scores[i1].fitness,
                (*currentGen)[scores[i2].index], scores[i2].fitness);
        }
    }
    else
    {
        // Just randomly select the rest population
        for (; nextGenomeIndex < config.numOrganismsInGeneration; ++nextGenomeIndex)
        {
            (*nextGen)[nextGenomeIndex] = (*currentGen)[randomi(randomGenerator)];
        }
    }

    generation.generationId++;
    generation.genomes = nextGen;
    genomeListBuffer = currentGen;
}

void NEAT::Mutate(Generation& g)
{
    RandomRealDistribution<float> randomf(-1.f, 1.f);
    NewlyAddedNodes newNodes;

    // Mutation weights
    for (auto& genome : *g.genomes)
    {
        if (genome.protect)
        {
            continue;
        }

        for (auto& elem : genome.connectionGenes)
        {
            ConnectionGene& connection = elem.second;
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
    for (auto& genome : *g.genomes)
    {
        if (genome.protect)
        {
            continue;
        }

        // Add a new node at random rate
        if (randomf(randomGenerator) <= config.nodeAdditionRate)
        {
            NodeAddedInfo node = AddNewNode(genome);
            if (node.newNode != invalidNodeGeneId)
            {
                newNodes[node.oldConnection].push_back(node);
            }
        }
    }

    EnsureUniqueGeneIndices(newNodes);

    for (auto& genome : *g.genomes)
    {
        if (genome.protect)
        {
            genome.protect = false;
            continue;
        }

        // Add a new connection at random rate
        if (randomf(randomGenerator) <= config.connectionAdditionRate)
        {
            AddNewConnection(genome);
        }
    }
}

// Make sure that the same topological changes have the same id
void NEAT::EnsureUniqueGeneIndices(const NewlyAddedNodes& newNodes)
{
    // Check duplicated newly added node genes
    for (auto& elem : newNodes)
    {
        auto& genomes = elem.second;
        const auto& info = genomes[0];

        NodeGeneId inNode = info.genome.connectionGenes[info.oldConnection].inNode;
        NodeGeneId outNode = info.genome.connectionGenes[info.oldConnection].outNode;

        for (size_t i = 1; i < genomes.size(); ++i)
        {
            const auto thisInfo = genomes[i];
            Genome& genome = genomes[i].genome;

            assert(genome.nodeLinks.find(info.newNode) == genome.nodeLinks.end());
            genome.nodeLinks[info.newNode] = genome.nodeLinks[thisInfo.newNode];
            genome.nodeLinks.erase(thisInfo.newNode);

            assert(genome.nodeLinks[info.newNode].incomings.size() == 1);
            assert(genome.nodeLinks[info.newNode].outgoings.size() == 1);
            genome.nodeLinks[info.newNode].incomings[0] = info.newConnection1;
            genome.nodeLinks[info.newNode].outgoings[0] = info.newConnection2;

            for (auto& innovId : genome.nodeLinks[outNode].incomings)
            {
                if (innovId == thisInfo.newConnection2)
                {
                    innovId = info.newConnection2;
                    break;
                }
            }
            for (auto& innovId : genome.nodeLinks[inNode].outgoings)
            {
                if (innovId == thisInfo.newConnection1)
                {
                    innovId = info.newConnection1;
                    break;
                }
            }

            assert(genome.connectionGenes.find(info.newConnection1) == genome.connectionGenes.end());
            genome.connectionGenes[info.newConnection1] = genome.connectionGenes[thisInfo.newConnection1];
            genome.connectionGenes[info.newConnection1].innovId = info.newConnection1;
            genome.connectionGenes[info.newConnection1].outNode = info.newNode;
            genome.connectionGenes.erase(thisInfo.newConnection1);

            assert(genome.connectionGenes.find(info.newConnection2) == genome.connectionGenes.end());
            genome.connectionGenes[info.newConnection2] = genome.connectionGenes[thisInfo.newConnection2];
            genome.connectionGenes[info.newConnection2].innovId = info.newConnection2;
            genome.connectionGenes[info.newConnection2].inNode = info.newNode;
            genome.connectionGenes.erase(thisInfo.newConnection2);

            innovationHistory.erase(NodePair{ inNode, thisInfo.newNode });
            innovationHistory.erase(NodePair{ thisInfo.newNode, outNode });

            assert(CheckSanity(genome));
        }
    }
}

void NEAT::Speciation(Generation& g, std::vector<Score>& scores, std::vector<int>& genomesToCopy)
{
    for (auto& species : g.species)
    {
        species.genomes.clear();
        if (species.stagnantGenerationCount == 0)
        {
            species.previousBestFitness = species.bestScore.fitness;
        }
        species.bestScore.fitness = 0.f;
    }

    for (size_t i = 0; i < config.numOrganismsInGeneration; ++i)
    {
        Genome& genome = (*g.genomes)[i];
        genome.species = invalidSpeciesId;
        for (auto& sp : generation.species)
        {
            const auto& representative = sp.representative;
            if (representative.connectionGenes.size() > 0)
            {
                if (CalculateDistance(genome, representative) < config.speciationDistThreshold)
                {
                    genome.species = sp.id;
                    sp.genomes.push_back(i);

                    if (sp.bestScore.fitness < scores[i].fitness)
                    {
                        sp.bestScore = scores[i];
                    }
                    break;
                }
            }
        }

        if (genome.species == invalidSpeciesId)
        {
            genome.species = currentSpeciesId;
            g.species.push_back(Species{ currentSpeciesId++ });
            g.species.back().genomes.push_back(i);
            g.species.back().bestScore = scores[i];
            g.species.back().representative = genome;
        }
    }

    for (auto itr = g.species.begin(); itr != g.species.end();)
    {
        auto& species = *itr;
        if (species.genomes.size() == 0)
        {
            itr = g.species.erase(itr);
            continue;
        }

        RandomIntDistribution<uint32_t> randomInt(0, species.genomes.size() - 1);
        int representativeIndex = species.genomes[randomInt(randomGenerator)];
        species.representative = (*g.genomes)[representativeIndex];

        bool extinct = false;
        if (species.previousBestFitness >= species.bestScore.fitness)
        {
            ++species.stagnantGenerationCount;
            if (species.stagnantGenerationCount >= 15)
            {
                extinct = true;
            }
        }

        if (!extinct)
        {
            for (int genomeIndex : species.genomes)
            {
                scores[genomeIndex].fitness /= (float)species.genomes.size();
            }

            if (species.genomes.size() >= 5)
            {
                genomesToCopy.push_back(species.bestScore.index);
                (*g.genomes)[species.bestScore.index].protect = true;
            }
        }
        else
        {
            for (int genomeIndex : species.genomes)
            {
                scores[genomeIndex].fitness = 0.0f;
            }
        }

        ++itr;
    }

    // Sort genomes by best score fitness
    std::sort(g.species.begin(), g.species.end(), [](const Species& s1, const Species& s2)
        {
            return s1.bestScore.fitness > s2.bestScore.fitness;
        });
}

// Calculate distance between two genomes based on their topologies and weights
float NEAT::CalculateDistance(const Genome& genome1, const Genome& genome2) const
{
    const auto& cGenes1 = genome1.connectionGenes;
    const auto& cGenes2 = genome2.connectionGenes;
    const size_t numConnectionsP1 = cGenes1.size();
    const size_t numConnectionsP2 = cGenes2.size();
    const size_t numConnections = numConnectionsP1 > numConnectionsP2 ? numConnectionsP1 : numConnectionsP2;
    size_t numMismatches = 0;
    size_t numMatches = 0;
    float weightDifference = 0.f;

    auto itr1 = cGenes1.begin();
    auto itr2 = cGenes2.begin();

    while (itr1 != cGenes1.end() && itr2 != cGenes2.end())
    {
        const ConnectionGene& cGene1 = itr1->second;
        const ConnectionGene& cGene2 = itr2->second;
        if (cGene1.innovId == cGene2.innovId)
        {
            weightDifference += std::fabs(cGene1.weight - cGene2.weight);
            ++numMatches;
            ++itr1;
            ++itr2;
        }
        else
        {
            ++numMismatches;
            if (cGene1.innovId < cGene2.innovId)
            {
                ++itr1;
            }
            else
            {
                ++itr2;
            }
        }
    }

    while (itr1 != cGenes1.end())
    {
        ++numMismatches;
        ++itr1;
    }
    while (itr2 != cGenes2.end())
    {
        ++numMismatches;
        ++itr2;
    }

    if (numMatches == 0)
    {
        return std::numeric_limits<float>::max();
    }
    else
    {
        return 2.f * (float)numMismatches / (float)numConnections + /*0.4f*/ 4.f * (weightDifference / (float)numMatches);
    }
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

    if (values.find(nodeId) != values.end())
    {
        return;
    }

    for (auto innovId : genome.nodeLinks.at(nodeId).incomings)
    {
        const auto connectionGene = GetConnectionGene(genome, innovId);
        assert(connectionGene != nullptr);

        if (!connectionGene->enabled) continue;

        auto incomingNodeId = connectionGene->inNode;
        bool alreadyEvaluating = false;

        if (values.find(incomingNodeId) == values.end())
        {
            //if (config.allowCyclicNetwork)
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
        }

        val += values[incomingNodeId] * connectionGene->weight;
    }

    ActivationFuncId activation;
    if (config.useGlobalActivationFunc)
    {
        activation = defaultActivationFuncId;
    }
    else
    {
        activation = generation.nodeGenes[nodeId].activationFuncId;
    }

    values[nodeId] = activationFuncs[activation](val);
}

void NEAT::SetupInitialNodeGenes()
{
    generation.nodeGenes.resize(3);

    generation.nodeGenes[0] = NodeGene{ NodeGeneType::Input, defaultActivationFuncId };
    generation.nodeGenes[1] = NodeGene{ NodeGeneType::Bias, defaultActivationFuncId };
    generation.nodeGenes[2] = NodeGene{ NodeGeneType::Output, defaultActivationFuncId };
}

// Create default genome for the initial generation
auto NEAT::CreateDefaultInitialGenome() const -> Genome
{
    Genome genome;

    NodeGeneId input = 0;
    NodeGeneId bias = 1;
    NodeGeneId output = 2;

    RandomRealDistribution<float> randomf(-1.f, 1.f);
    {
        InnovationId innovId = 0;
        ConnectionGene gene{ innovId, input, output, randomf(randomGenerator), true};
        genome.nodeLinks[input].outgoings.push_back(innovId);
        genome.nodeLinks[output].incomings.push_back(innovId);
        genome.connectionGenes[innovId] = gene;
    }
    {
        InnovationId innovId = 1;
        ConnectionGene gene{ innovId, bias, output, randomf(randomGenerator), true };
        genome.nodeLinks[bias].outgoings.push_back(innovId);
        genome.nodeLinks[output].incomings.push_back(innovId);
        genome.connectionGenes[innovId] = gene;
    }

    return genome;
}

auto NEAT::SelectRandomeGenome() -> Genome&
{
    RandomIntDistribution<uint32_t> distribution(0, (*generation.genomes).size() - 1);
    return (*generation.genomes)[distribution(randomGenerator)];
}

auto NEAT::SelectRandomNodeGene(const std::vector<NodeGeneId>& genes) const -> NodeGeneId
{
    RandomIntDistribution<NodeGeneId> distribution(0, genes.size() - 1);
    return genes[distribution(randomGenerator)];
}

auto NEAT::SelectRandomNodeGene(const Genome& genome) const -> NodeGeneId
{
    RandomIntDistribution<NodeGeneId> distribution(0, genome.nodeLinks.size() - 1);
    return std::next(std::begin(genome.nodeLinks), distribution(randomGenerator))->first;
}

auto NEAT::SelectRandomConnectionGene(const std::vector<InnovationId>& genes) const -> InnovationId
{
    RandomIntDistribution<InnovationId> distribution(0, genes.size() - 1);
    return *std::next(std::begin(genes), distribution(randomGenerator));
}

auto NEAT::SelectRandomConnectionGene(const Genome& genome) const -> InnovationId
{
    RandomIntDistribution<InnovationId> distribution(0, genome.connectionGenes.size() - 1);
    return std::next(std::begin(genome.connectionGenes), distribution(randomGenerator))->second.innovId;
}

auto NEAT::GetConnectionGene(Genome& genome, InnovationId innovId) const -> ConnectionGene*
{
    return genome.connectionGenes.find(innovId) != genome.connectionGenes.end() ?
        &genome.connectionGenes[innovId] :
        nullptr;
}

auto NEAT::GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*
{
    return genome.connectionGenes.find(innovId) != genome.connectionGenes.end() ?
        &genome.connectionGenes.at(innovId) :
        nullptr;
}

bool NEAT::CheckSanity(const Genome& genome) const
{
#ifdef _DEBUG
    if (!config.enableSanityCheck)
    {
        return true;
    }

    // Check if there's no duplicated connection genes for the same in and out nodes
    {
        const auto& conns = genome.connectionGenes;
        for (auto itr1 = conns.begin(); itr1 != conns.end(); ++itr1)
        {
            auto itr2 = itr1;
            for (++itr2; itr2 != conns.end(); ++itr2)
            {
                const auto& c1 = itr1->second;
                const auto& c2 = itr2->second;
                if (c1.inNode == c2.inNode && c1.outNode == c2.outNode)
                {
                    return false;
                }
            }
        }
    }

    if (!config.allowCyclicNetwork)
    {
        for (const auto con : genome.connectionGenes)
        {
            if (con.second.enabled && !CheckCyclic(genome, con.second.inNode, con.second.outNode))
            {
                return false;
            }
        }
    }

#endif
    return true;
}


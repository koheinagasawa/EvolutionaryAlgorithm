#include "NEAT.h"

#include <bitset>
#include <algorithm>
#include <stack>
#include <fstream>
#include <iostream>

std::default_random_engine NEAT::s_randomGenerator(1);

#define TEST_PREVENT_TO_ADD_NEW_CONNECTION_TO_DEADEND_HIDDEN_NODE 1

// Initialize NEAT
// This has to be called before gaining any generations
// Returns the initial generation
auto NEAT::Initialize(const Configration& configIn) -> const Generation &
{
    m_isInitialized = false;

    // Set configuration
    m_config = configIn;

    // Clear any existing data
    Reset();

    // Sanity checks
    if (m_config.m_numOrganismsInGeneration == 0)
    {
        // TODO: Add some useful warning message here
        return m_generation;
    }
    if (m_config.m_activateFunctions.size() == 0 || m_config.m_activateFunctions.size() <= m_config.m_defaultActivateFunctionId)
    {
        // TODO: Add some useful warning message here
        return m_generation;
    }

    // Set activations
    m_activationFuncs = m_config.m_activateFunctions;
    m_defaultActivationFuncId = m_config.m_defaultActivateFunctionId;

    // Set up node used for the initial network
    SetupInitialNodeGenes();

    // Populate the first generation with default genomes
    for (uint32_t i = 0; i < m_config.m_numOrganismsInGeneration; ++i)
    {
        AccessGenome(i) = CreateDefaultInitialGenome();
    }

    m_isInitialized = true;

    return m_generation;
}

// Reset all the state in NEAT
// After calling this function, Initialize has to be called in order to run NEAT again
void NEAT::Reset()
{
    // Reset generation
    {
        GenomeList genomes = m_generation.m_genomes;
        m_generation = Generation();
        m_generation.m_genomes = genomes;
    }

    auto allocateGenomesBuffer = [this](GenomeList& buffer)
    {
        buffer = std::make_shared<std::vector<Genome>>();
        buffer->resize(m_config.m_numOrganismsInGeneration);
    };

    // Allocate buffer for genomes
    {
        if (!m_generation.m_genomes)
        {
            allocateGenomesBuffer(m_generation.m_genomes);
        }

        if (!m_nextGenGenomesBuffer)
        {
            allocateGenomesBuffer(m_nextGenGenomesBuffer);
        }
    }

    // Reset global ids
    m_currentInnovationId = 0;
    m_currentSpeciesId = 0;

    // Allocate memory for scores
    m_scores.resize(m_config.m_numOrganismsInGeneration);
}

// Gain the new generation
auto NEAT::GetNewGeneration(bool printFitness) -> const Generation&
{
    if (m_isInitialized)
    {
        GenerateNewGeneration(printFitness);
    }
    return m_generation;
}

// Return the current generation
auto NEAT::GetCurrentGeneration() -> const Generation&
{
    return m_generation;
}

// Print fitness of each genome in the current generation
void NEAT::PrintFitness() const
{
    std::cout << "\nGeneration " << m_generation.m_generationId << std::endl;
    float sum = 0;
    const int population = GetNumGenomes();
    float maxFitness = 0;
    int bestGenomeId = -1;
    int bestSpeciesId = -1;

    for (int i = 0; i < population; ++i)
    {
        const Genome& genome = GetGenome(i);
        const float fitness = Evaluate(genome);

        sum += fitness;
        if (fitness > maxFitness)
        {
            maxFitness = fitness;
            bestGenomeId = i;
            bestSpeciesId = genome.m_species;
        }
    }

    const Species* bestSpecies = nullptr;
    for (const auto& sp : m_generation.m_species)
    {
        if (sp.m_id == bestSpeciesId)
        {
            bestSpecies = &sp;
            break;
        }
    }


    std::cout << "Average fitness over " << population << " organisms: " << sum / (float)population << std::endl;
    std::cout << "Maximum fitness: Species " << bestSpeciesId << ": Genome " << bestGenomeId << " - Score: " << maxFitness << std::endl;
    std::cout << "Number of species: " << m_generation.m_species.size() << std::endl;
    std::cout << "Number of genomes in the best species: " << bestSpecies->m_scores.size() << std::endl;
}

// Serialize generation as a json file
void NEAT::SerializeGeneration(const char* fileName) const
{
    using namespace std;
    ofstream json(fileName);

    if (m_generation.m_species.size() == 0)
    {
        // Generation is empty.
        json << "{}" << endl;
        json.close();

        // TODO: Add some useful warning message here
        return;
    }

    int tabLevel = 0;
    auto printTabs = [&json, &tabLevel]()
    {
        for (int i = 0; i < tabLevel; ++i){ json << "    "; }
    };

    struct Scope
    {
        Scope(int& tabLevel) : m_tabLevel(tabLevel) { ++m_tabLevel; }
        ~Scope() { --m_tabLevel; }
        int& m_tabLevel;
    };

    json << "{" << endl;
    {
        Scope s(tabLevel);

        printTabs(); json << "\"GenerationId\" : " << m_generation.m_generationId << "," << endl;

        printTabs(); json << "\"Species\" : [" << endl;
        {
            for (int i = 0; i < (int)m_generation.m_species.size(); ++i)
            {
                const Species& spc = m_generation.m_species[i];
                printTabs(); json << "{" << endl;
                {
                    Scope s(tabLevel);
                    printTabs(); json << "\"Id\" : " << spc.m_id << "," << endl;
                    printTabs(); json << "\"BestScore\" : " << spc.m_bestScore.m_fitness << "," << endl;
                    printTabs(); json << "\"BestScoreGenomeIndex\" : " << spc.m_bestScore.m_index << endl;
                }
                printTabs(); json << "}";

                if (i != (int)m_generation.m_species.size() - 1)
                {
                    json << "," << endl;
                }
                else
                {
                    json << endl;
                }
            }
        }
        printTabs(); json << "]," << endl;

        printTabs(); json << "\"Genomes\" : [" << endl;
        {
            int index = 0;
            for (int i = 0; i < GetNumGenomes(); ++i)
            {
                const Genome& genome = GetGenome(i);

                printTabs(); json << "{" << endl;
                {
                    Scope s(tabLevel);
                    printTabs(); json << "\"Index\" : " << index << "," << endl;
                    printTabs(); json << "\"SpeciesId\" : " << genome.m_species << "," << endl;

                    printTabs(); json << "\"Nodes\" : [" << endl;
                    {
                        int i = 0;
                        for (auto& elem : genome.m_nodeLinks)
                        {
                            printTabs(); json << "{" << endl;
                            {
                                Scope s(tabLevel);
                                NodeGeneId id = elem.first;
                                printTabs(); json << "\"Id\" : " << id << "," << endl;
                                const char* type = nullptr;
                                switch (m_generation.m_nodeGenes[id].m_type)
                                {
                                case NodeGeneType::Input:
                                    type = "\"INPUT\"";
                                    break;
                                case NodeGeneType::Output:
                                    type = "\"OUTPUT\"";
                                    break;
                                case NodeGeneType::Hidden:
                                    type = "\"HIDDEN\"";
                                    break;
                                case NodeGeneType::Bias:
                                    type = "\"BIAS\"";
                                    break;
                                default:
                                    break;
                                }
                                printTabs(); json << "\"Type\" : " << type << endl;
                            }
                            printTabs(); json << "}";
                            ++i;
                            if (i != (int)genome.m_nodeLinks.size())
                            {
                                json << "," << endl;
                            }
                            else
                            {
                                json << endl;
                            }
                        }
                    }
                    printTabs(); json << "]," << endl;

                    printTabs(); json << "\"Connections\" : [" << endl;
                    {
                        int i = 0;
                        for (auto& elem : genome.m_connectionGenes)
                        {
                            printTabs(); json << "{" << endl;
                            {
                                Scope s(tabLevel);
                                const ConnectionGene& cg = elem.second;
                                printTabs(); json << "\"InnovationId\" : " << cg.m_innovId << "," << endl;
                                printTabs(); json << "\"InNode\" : " << cg.m_inNode << "," << endl;
                                printTabs(); json << "\"OutNode\" : " << cg.m_outNode << "," << endl;
                                printTabs(); json << "\"Weight\" : " << cg.m_weight << "," << endl;
                                printTabs(); json << "\"Enabled\" : ";
                                cg.m_enabled ? (json << "true" << endl) : (json << "false" << endl);
                            }
                            printTabs(); json << "}";
                            ++i;
                            if (i != (int)genome.m_connectionGenes.size())
                            {
                                json << "," << endl;
                            }
                            else
                            {
                                json << endl;
                            }
                        }
                    }
                    printTabs(); json << "]" << endl;
                }
                printTabs(); json << "}";
                ++index;
                if (i != GetNumGenomes() - 1)
                {
                    json << "," << endl;
                }
                else
                {
                    json << endl;
                }
            }
        }
        printTabs(); json << "]" << endl;
    }
    json << "}" << endl;
    json.close();
}

// Increment innovation id and return the new id
auto NEAT::GetNewInnovationId() -> InnovationId
{
    // TODO: Make it thread safe
    return m_currentInnovationId++;
}

// Create a new node of a type
auto NEAT::CreateNewNode(NodeGeneType type) -> NodeGeneId
{
    NodeGene node{ type };

    if (m_config.m_useGlobalActivationFunc)
    {
        node.m_activationFuncId = m_defaultActivationFuncId;
    }
    else
    {
        // Set a random activation function
        RandomIntDistribution<uint32_t> distribution(0, m_activationFuncs.size());
        node.m_activationFuncId = distribution(s_randomGenerator);
    }

    NodeGeneId nodeGeneId = m_generation.m_nodeGenes.size();
    m_generation.m_nodeGenes.push_back(node);

    return nodeGeneId;
}

// Add a new node at a random connection
auto NEAT::AddNewNode(Genome& genome) -> NodeAddedInfo
{
    // First, collect all connections where we can add new node
    std::vector<InnovationId> availableConnections;
    {
        availableConnections.reserve(genome.GetNumConnections());

        for (const auto& elem : genome.m_connectionGenes)
        {
            const ConnectionGene& c = elem.second;
            // We can add a new node to connections enabled and not connected to a bias node
            if (c.m_enabled && GetNodeGene(c.m_inNode).m_type != NodeGeneType::Bias)
            {
                availableConnections.push_back(c.m_innovId);
            }
        }

        // Return when there's no available connection to add a new node
        if (availableConnections.size() == 0)
        {
            // TODO: Output a useful message
            return NodeAddedInfo{genome, s_invalidNodeGeneId};
        }
    }

    // Choose a random connection gene from the available ones
    const InnovationId innovId = SelectRandomConnectionGene(availableConnections);
    ConnectionGene* c = GetConnectionGene(genome, innovId);
    assert(c);

    // Get values of this connection gene before we modify the genome
    // (the pointer to gene can become invalid after making any modification to the genome)
    const NodeGeneId inNode = c->m_inNode;
    const NodeGeneId outNode = c->m_outNode;
    const float weight = c->m_weight;

    // Disable existing connection
    genome.DisableConnection(innovId);

    // Create a new node
    NodeGeneId newNodeId = CreateNewNode(NodeGeneType::Hidden);
    genome.AddNode(newNodeId);

    // Create a new connection between the inNode and the new node
    InnovationId newCon1 = Connect(genome, inNode, newNodeId, 1.f);

    // Create a new connection between the new node and the outNode
    InnovationId newCon2 = Connect(genome, newNodeId, outNode, weight);

    assert(CheckSanity(genome));

    return NodeAddedInfo{ genome, newNodeId, innovId, newCon1, newCon2 };
}

// Connect the two nodes and assign the weight
auto NEAT::Connect(Genome& genome, NodeGeneId inNode, NodeGeneId outNode, float weight) -> InnovationId
{
    assert(genome.HasNode(inNode));
    assert(genome.HasNode(outNode));

    InnovationId innovId;
    {
        // [TODO] : Treating connections with the same inNode and outNode as the same innovation
        //          might not be along with the original paper. It says innovation ID is chronological
        //          information, then maybe connections created at different timing should be handled as
        //          different innovations even if they have the same inNode and outNode.
        //          But doing so can lead to diversity explosion easily. Also we will need to decide what to do
        //          when two connections with different innovation ids but the same position have inherited
        //          during cross over.

        // Register this connection to the history
        NodePair pair(inNode, outNode);
        // Check if the equivalent connection has been added in the past
        if (m_innovationHistory.find(pair) == m_innovationHistory.end())
        {
            // This is a completely new connection. Assign a new innovation Id.
            innovId = GetNewInnovationId();
            m_innovationHistory[pair] = innovId;
        }
        else
        {
            // An equivalent connection has been added before. Use that innovation Id.
            innovId = m_innovationHistory.at(pair);
        }

        assert(!genome.HasConnection(innovId));
    }

    // Setup new connection
    ConnectionGene newConnection;
    {
        newConnection.m_inNode = inNode;
        newConnection.m_outNode = outNode;
        newConnection.m_innovId = innovId;
        newConnection.m_enabled = true;
        newConnection.m_weight = weight;
    }
    
    // Add the connection
    genome.AddConnection(newConnection);

    return innovId;
}

// Add a new connection between random two nodes without allowing cyclic network
// If allowCyclic is false, direction of the new connection is guaranteed to be one direction 
// (distance from in node to an input node is smaller than the one of out node)
void NEAT::AddNewConnection(Genome& genome, bool allowCyclic)
{
    const size_t numNodes = genome.GetNumNodes();

    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> outNodeCandidates;
    {
        for (const auto& elem : genome.m_nodeLinks)
        {
            const NodeGeneId nodeId = elem.first;
            const NodeGene& node = GetNodeGene(nodeId);

            // We can create a new connection leading into either output nodes or hidden nodes
            if (node.m_type == NodeGeneType::Output || node.m_type == NodeGeneType::Hidden)
            {

#if TEST_PREVENT_TO_ADD_NEW_CONNECTION_TO_DEADEND_HIDDEN_NODE
                // Select only hidden nodes which are not dead end (no enabled outgoing connections)
                // [TODO]: Verify this additional condition for hidden node which is not stated in the original paper
                if (node.m_type == NodeGeneType::Hidden && elem.second.m_numEnabledOutgoings == 0)
                {
                    continue;
                }
#endif

                // Check a rare case that the node is already connected to all the existing node
                if (genome.m_nodeLinks.at(nodeId).m_incomings.size() < numNodes)
                {
                    outNodeCandidates.push_back(nodeId);
                }
            }
        }
    }

    // Return when there's no available node to add a new connection
    if (outNodeCandidates.size() == 0)
    {
        // TODO: Output a useful message message
        return;
    }

    // Randomize the order of available nodes in the array
    std::random_shuffle(outNodeCandidates.begin(), outNodeCandidates.end());

    for (auto outNodeId : outNodeCandidates)
    {
        // Then collect all node genes which are not connected to the outNode already
        std::vector<NodeGeneId> inNodeCandidates;
        {
            // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
            std::vector<int> connectedFlag;
            connectedFlag.resize(((int)m_generation.m_nodeGenes.size() / sizeof(int)) + 1, 0);

            // Mark nodes connected to outNode
            for (auto innovId : genome.GetIncommingConnections(outNodeId))
            {
                const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
                assert(cgene);
                const NodeGeneId nodeId = cgene->m_inNode;
                connectedFlag[nodeId / sizeof(int)] |= (1 << (nodeId % sizeof(int)));
            }

            // Find all nodes not connected to outNode
            for (const auto& elem : genome.m_nodeLinks)
            {
                const NodeGeneId nodeId = elem.first;
                const NodeGene& node = GetNodeGene(nodeId);

                if (nodeId == outNodeId) continue;

                // We can create a new connection leading from either input, bias and hidden nodes
                if (node.m_type == NodeGeneType::Input ||
                    node.m_type == NodeGeneType::Bias ||
                    node.m_type == NodeGeneType::Hidden)
                {
#if TEST_PREVENT_TO_ADD_NEW_CONNECTION_TO_DEADEND_HIDDEN_NODE
                    // Select only hidden nodes which are not dead end (no enabled incoming connections)
                    // [TODO]: Verify this additional condition for hidden node which is not stated in the original paper
                    if (node.m_type == NodeGeneType::Hidden && elem.second.m_numEnabledIncomings == 0)
                    {
                        continue;
                    }
#endif

                    // Check if the node is already connected to the out node
                    if (((connectedFlag[nodeId / sizeof(int)] >> (nodeId % sizeof(int))) & 1) == 0)
                    {
                        // Check cyclic network
                        if (allowCyclic || CanAddConnectionWithoutCyclic(genome, nodeId, outNodeId))
                        {
                            inNodeCandidates.push_back(nodeId);
                        }
                    }
                }
            }
        }

        if (inNodeCandidates.size() == 0)
        {
            // No available node where we can create a forward connection found
            // Continue and try a new output node candidate
            continue;
        }

        // Select a random node from the available ones as outNode
        NodeGeneId inNodeId = SelectRandomNodeGene(inNodeCandidates);;

        // Add a new connection
        Connect(genome, inNodeId, outNodeId, GetRandomWeight());

        assert(CheckSanity(genome));

        return;
    }

    // No available nodes were found
    // TODO: Output some useful message
}

// Return false if adding a connection between srcNode to targetNode makes the network cyclic
bool NEAT::CanAddConnectionWithoutCyclic(const Genome& genome, NodeGeneId srcNode, NodeGeneId targetNode) const
{
    if (srcNode == targetNode)
    {
        return false;
    }

    // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
    std::vector<int> flag;
    flag.resize((int)m_generation.m_nodeGenes.size() / sizeof(int) + 1, 0);

    std::stack<NodeGeneId> stack;
    stack.push(srcNode);
    while (!stack.empty())
    {
        const NodeGeneId node = stack.top();
        stack.pop();

        for (const InnovationId innovId : genome.GetIncommingConnections(node))
        {
            const ConnectionGene* con = GetConnectionGene(genome, innovId);
            assert(con);

            if (!con->m_enabled) continue;

            const NodeGeneId inNode = con->m_inNode;

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
auto NEAT::CrossOver(const Genome& genome1, float fitness1, const Genome& genome2, float fitness2) const -> Genome
{
    const Genome* parent1 = &genome1;
    const Genome* parent2 = &genome2;

    // Make sure that parent1 has higher or equal fitness than parent2
    const bool sameFitness = fitness1 == fitness2;
    if (fitness1 < fitness2)
    {
        parent1 = &genome2;
        parent2 = &genome1;
    }

    // Create a new genome
    Genome child;

    // Try to add a consecutive incompatible region to the child genome as a batch
    auto TryAddIncompatibleRegion = [this, &child](
        const NodeGeneId incompatibleNode,
        const Genome* base,
        const Genome* other)
    {
        IncompatibleRegion ir;
        GetIncompatibleRegionRecursive(incompatibleNode, base, other, ir);

        std::vector<NodeGeneId> nodesAdded;
        nodesAdded.reserve(ir.m_nodes.size());
        for (NodeGeneId node : ir.m_nodes)
        {
            if (child.m_nodeLinks.find(node) == child.m_nodeLinks.end())
            {
                child.m_nodeLinks.insert({ node, Genome::Links() });
                nodesAdded.push_back(node);
            }
        }

        std::vector<InnovationId> connectionsAdded;
        connectionsAdded.reserve(ir.m_connections.size());
        bool cyclic = false;
        for (auto itr = ir.m_connections.begin(); itr != ir.m_connections.end(); ++itr)
        {
            if (child.m_connectionGenes.find(*itr) == child.m_connectionGenes.end())
            {
                const ConnectionGene& c = base->m_connectionGenes.at(*itr);
                if (m_config.m_allowCyclicNetwork || CanAddConnectionWithoutCyclic(child, c.m_inNode, c.m_outNode))
                {
                    ConnectionGene newCon = c;
                    child.m_connectionGenes[newCon.m_innovId] = newCon;
                    child.m_nodeLinks[c.m_outNode].m_incomings.push_back(c.m_innovId);
                    child.m_nodeLinks[c.m_inNode].m_outgoings.push_back(c.m_innovId);
                    if (newCon.m_enabled)
                    {
                        child.m_nodeLinks[c.m_outNode].m_numEnabledIncomings++;
                        child.m_nodeLinks[c.m_inNode].m_numEnabledOutgoings++;
                    }
                    connectionsAdded.push_back(c.m_innovId);
                }
                else
                {
                    // Abort adding this region when it causes cyclic network
                    cyclic = true;
                    break;
                }
            }
        }

        if (cyclic)
        {
            for (NodeGeneId node : nodesAdded)
            {
                child.m_nodeLinks.erase(node);
            }

            for (InnovationId iid : connectionsAdded)
            {
                const ConnectionGene& c = child.m_connectionGenes[iid];
                {
                    if (child.m_nodeLinks.find(c.m_inNode) != child.m_nodeLinks.end())
                    {
                        Genome::Links& links = child.m_nodeLinks[c.m_inNode];
                        for (auto i = links.m_outgoings.begin(); i != links.m_outgoings.end(); ++i)
                        {
                            if (*i == iid)
                            {
                                links.m_outgoings.erase(i);
                                if (c.m_enabled)
                                {
                                    links.m_numEnabledOutgoings--;
                                }
                                break;
                            }
                        }
                    }
                }
                {
                    if (child.m_nodeLinks.find(c.m_outNode) != child.m_nodeLinks.end())
                    {
                        Genome::Links& links = child.m_nodeLinks[c.m_outNode];
                        for (auto i = links.m_incomings.begin(); i != links.m_incomings.end(); ++i)
                        {
                            if (*i == iid)
                            {
                                links.m_incomings.erase(i);
                                if (c.m_enabled)
                                {
                                    links.m_numEnabledIncomings--;
                                }
                                break;
                            }
                        }
                    }
                }

                child.m_connectionGenes.erase(iid);
            }
        }
    };

    // Try to add the given connection to the child genome
    auto TryAddConnection = [this, &child, &TryAddIncompatibleRegion](
        const ConnectionGene& connection,
        const Genome* base,
        const Genome* other,
        bool enable)
    {
        // Check if we've already added this connection
        if (child.HasConnection(connection.m_innovId))
        {
            return;
        }

        const NodeGeneId inNode = connection.m_inNode;
        const NodeGeneId outNode = connection.m_outNode;

        assert(base->HasNode(inNode));
        assert(base->HasNode(outNode));

        // If Either in or out node doesn't exist in the other parent,
        // we try to add the entire consecutive incompatible regions as a batch instead of just a single connection
        bool hasIncompatibleNode = false;
        if (!other->HasNode(inNode))
        {
            TryAddIncompatibleRegion(inNode, base, other);
            hasIncompatibleNode = true;
        }
        if (!other->HasNode(outNode))
        {
            TryAddIncompatibleRegion(outNode, base, other);
            hasIncompatibleNode = true;
        }
        if (hasIncompatibleNode)
        {
            return;
        }

        // Make sure that child genome has both in and out nodes
        if (!child.HasNode(inNode))  child.AddNode(inNode);
        if (!child.HasNode(outNode)) child.AddNode(outNode);

        // Force to disable connection when it causes cyclic network
        if (!m_config.m_allowCyclicNetwork && !CanAddConnectionWithoutCyclic(child, inNode, outNode))
        {
            enable = false;
        }

        // Add new connection
        ConnectionGene newCon = connection;
        newCon.m_enabled = enable;
        child.AddConnection(newCon);
    };

    // Inherit connection genes from the two parents
    RandomIntDistribution<int> randomBinary(0, 1);

    const auto& cGenes1 = parent1->m_connectionGenes;
    const auto& cGenes2 = parent2->m_connectionGenes;

    auto itr1 = cGenes1.begin();
    auto itr2 = cGenes2.begin();

    // Select each gene from either parent based on gene's innovation id and parents' fitness
    // Note that cGenes1/2 are sorted by innovation id
    while (itr1 != cGenes1.end() && itr2 != cGenes2.end())
    {
        const ConnectionGene& cGene1 = itr1->second;
        const ConnectionGene& cGene2 = itr2->second;
        const ConnectionGene* geneToInherit = nullptr;
        bool fromP1 = true;
        bool enabled = true;

        // When two genes have the same innovation id, take one gene randomly from either parent
        // Or if two genes have the same fitness values, always take one gene randomly from either parent regardless of innovation id
        if (cGene1.m_innovId == cGene2.m_innovId)
        {
            geneToInherit = randomBinary(s_randomGenerator) ? &cGene1 : &cGene2;

            // The gene of the new genome could be disable when the gene of either parent is disabled
            if ((!cGene1.m_enabled || !cGene2.m_enabled) && GetRandomProbability() < m_config.m_geneDisablingRate)
            {
                enabled = false;
            }

            ++itr1;
            ++itr2;
        }
        else
        {
            // If this gene exists only in parent1, inherit from parent1
            if (cGene1.m_innovId < cGene2.m_innovId)
            {
                geneToInherit = !sameFitness || randomBinary(s_randomGenerator) ? &cGene1 : nullptr;
                ++itr1;
            }
            // If this gene exists only in parent2, don't inherit it
            else
            {
                geneToInherit = sameFitness && randomBinary(s_randomGenerator) ? &cGene2 : nullptr;
                fromP1 = false;
                ++itr2;
            }
        }

        if (geneToInherit)
        {
            // Add connection gene to the new child
            TryAddConnection(
                *geneToInherit,
                fromP1 ? parent1 : parent2,
                fromP1 ? parent2 : parent1,
                enabled);
        }
    }

    // Add remaining genes
    {
        auto AddRemainingGenes = [&TryAddConnection, &randomBinary](
            bool randomize,
            const Genome* base,
            const Genome* other,
            ConnectionGeneList::const_iterator& itr)
        {
            while (itr != base->m_connectionGenes.end())
            {
                if (!randomize || randomBinary(s_randomGenerator))
                {
                    const ConnectionGene& gene = itr->second;
                    TryAddConnection(gene, base, other, gene.m_enabled);
                }
                ++itr;
            }
        };

        // Add remaining genes from parent1
        AddRemainingGenes(sameFitness, parent1, parent2, itr1);

        // Add remaining genes from parent2 if fitness is the same
        if (sameFitness)
        {
            AddRemainingGenes(true, parent2, parent1, itr2);
        }
    }

    // [TODO] Investigate why. Maybe just printFitness function would access to it?
    child.m_species = parent1->m_species;

    assert(CheckSanity(child));

    return child;
}

void NEAT::GetIncompatibleRegionRecursive(NodeGeneId current, const Genome* base, const Genome* other, IncompatibleRegion& incompatible) const
{
    assert(base->m_nodeLinks.find(current) != base->m_nodeLinks.end());

    if (incompatible.m_nodes.find(current) == incompatible.m_nodes.end())
    {
        incompatible.m_nodes.insert(current);
        if (other->m_nodeLinks.find(current) == other->m_nodeLinks.end())
        {
            const Genome::Links& links = base->m_nodeLinks.at(current);
            for (InnovationId iid : links.m_incomings)
            {
                incompatible.m_connections.insert(iid);
                GetIncompatibleRegionRecursive(base->m_connectionGenes.at(iid).m_inNode, base, other, incompatible);
            }
            for (InnovationId iid : links.m_outgoings)
            {
                incompatible.m_connections.insert(iid);
                GetIncompatibleRegionRecursive(base->m_connectionGenes.at(iid).m_outNode, base, other, incompatible);
            }
        }
    }
}

// Implementation of generating a new generation
void NEAT::GenerateNewGeneration(bool printFitness)
{
    Mutate();

    if (m_config.m_diversityProtection == DiversityProtectionMethod::Speciation)
    {
        for (uint32_t i = 0; i < m_config.m_numOrganismsInGeneration; ++i)
        {
            m_scores[i] = Score{ Evaluate(GetGenome(i)), 0.f, i };
        }

        Speciation();
    }
    else if (m_config.m_diversityProtection == DiversityProtectionMethod::MorphologicalInnovationProtection)
    {
        // Not implemented yet
        assert(0);
    }

    SelectGenomes();

    if (printFitness)
    {
        // Sort species by best score fitness
        std::sort(m_generation.m_species.begin(), m_generation.m_species.end(), [](const Species& s1, const Species& s2)
        {
            return s1.m_bestScore.m_fitness > s2.m_bestScore.m_fitness;
        });

        PrintFitness();
    }
}

void NEAT::SelectGenomes()
{
    std::vector<Score> genomesToInherit;
    genomesToInherit.reserve(GetNumGenomes());

    float adjustedScoreSum = 0;
    for (const auto& s : m_generation.m_species)
    {
        adjustedScoreSum += s.m_adjustedTotalScore;
    }

    RandomRealDistribution<float> genomeSelector(0, adjustedScoreSum);
    auto getGenome = [](float f, const std::vector<Score>& scores) -> const Score&
    {
        float currentSum = 0;
        for (const auto& score : scores)
        {
            currentSum += score.m_adjustedFitness;
            if (currentSum >= f)
            {
                return score;
            }
        }

        return scores.back();
    };

    bool canCrossOver = false;
    if (m_config.m_enableCrossOver)
    {
        for (const auto& sp : m_generation.m_species)
        {
            if (sp.m_scores.size() > 1)
            {
                canCrossOver = true;
                break;
            }
        }
    }

    std::vector<Genome> newGenomes;

    // Distribute population over the species
    {
        const int totalPopulation = int(GetNumGenomes() * (1.0f - m_config.m_interSpeciesMatingRate));
        const float invAdjustedScoreSum = 1.0f / adjustedScoreSum;
        for (const Species& sp : m_generation.m_species)
        {
            // Remove lowest fitness genomes from selection
            const int numGenomesToRemove = sp.m_scores.size() / 5;
            float totalScore = sp.m_adjustedTotalScore;
            for (int i = 0; i < numGenomesToRemove; ++i)
            {
                totalScore -= sp.m_scores[sp.m_scores.size() - 1 - i].m_adjustedFitness;
            }

            RandomRealDistribution<float> genomeInSpeciesSelector(0, totalScore);

            const int spPopulation = (int)(totalPopulation * sp.m_adjustedTotalScore * invAdjustedScoreSum);
            const int numOrgsToCopy = canCrossOver ? int(spPopulation * (1.f - m_config.m_crossOverRate)) : spPopulation;

            if (/*numOrgsToCopy > 0 && */sp.ShouldProtectBest())
            {
                genomesToInherit.push_back(sp.m_scores[0]);
            }

            // Just copy high score genomes to the next generation
            for (int i = (int)genomesToInherit.size(); i < numOrgsToCopy; ++i)
            {
                genomesToInherit.push_back(getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores));
            }

            for (int i = (int)genomesToInherit.size(); i < spPopulation; ++i)
            {
                if (sp.m_scores.size() == 1)
                {
                    genomesToInherit.push_back(sp.m_scores[0]);
                    continue;
                }

                // Select random two genomes
                const Score* g1 = &getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores);
                const Score* g2 = &getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores);

                while (g1 == g2)
                {
                    g2 = &getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores);
                }

                // Ensure the genome at i1 has a higher fitness
                if (g1->m_fitness < g2->m_fitness)
                {
                    std::swap(g1, g2);
                }

                // Cross over
                newGenomes.push_back(CrossOver(GetGenome(g1->m_index), g1->m_fitness, GetGenome(g2->m_index), g2->m_fitness));
                const float fitness = Evaluate(newGenomes.back());
                if (fitness >= sp.m_bestScore.m_fitness && sp.m_scores.size() >= 5)
                {
                    newGenomes.back().m_protect = true;
                    if (fitness > sp.m_bestScore.m_fitness)
                    {
                        AccessGenome(sp.m_bestScore.m_index).m_protect = false;
                    }
                }
                genomesToInherit.push_back(Score{ fitness, -1.f, newGenomes.size() - 1 });
            }
        }

        // Cross over remaining population
        if (m_generation.m_species.size() == 1)
        {
            for (int i = (int)genomesToInherit.size(); i < GetNumGenomes(); ++i)
            {
                RandomRealDistribution<float> genomeSelector(0, adjustedScoreSum);
                // Select random two genomes
                const Score* g1 = &getGenome(genomeSelector(s_randomGenerator), m_scores);
                const Score* g2 = &getGenome(genomeSelector(s_randomGenerator), m_scores);

                while (g1 == g2)
                {
                    g2 = &getGenome(genomeSelector(s_randomGenerator), m_scores);
                }

                // Ensure the genome at i1 has a higher fitness
                if (g1->m_fitness < g2->m_fitness)
                {
                    std::swap(g1, g2);
                }

                // Cross over
                newGenomes.push_back(CrossOver(GetGenome(g1->m_index), g1->m_fitness, GetGenome(g2->m_index), g2->m_fitness));
                const float fitness = Evaluate(newGenomes.back());
                if (fitness >= m_generation.m_species[0].m_bestScore.m_fitness && m_generation.m_species[0].m_scores.size() >= 5)
                {
                    newGenomes.back().m_protect = true;
                    if (fitness > m_generation.m_species[0].m_bestScore.m_fitness)
                    {
                        AccessGenome(m_generation.m_species[0].m_bestScore.m_index).m_protect = false;
                    }
                }
                genomesToInherit.push_back(Score{ fitness, -1.f, newGenomes.size() - 1 });
            }
        }
        // Perform inter species cross over
        else
        {
            for (int i = (int)genomesToInherit.size(); i < GetNumGenomes(); ++i)
            {
                RandomRealDistribution<float> speciesSelector(0, adjustedScoreSum);
                auto getSpecies = [this](float f)
                {
                    float currentSum = 0;
                    for (int i = 0; i < (int)m_generation.m_species.size(); ++i)
                    {
                        const auto& species = m_generation.m_species[i];
                        currentSum += species.m_adjustedTotalScore;
                        if (currentSum > f)
                        {
                            return i;
                        }
                    }

                    return (int)m_generation.m_species.size() - 1;
                };

                const auto* sp1 = &m_generation.m_species[getSpecies(speciesSelector(s_randomGenerator))];
                if (sp1->m_scores.size() == 1)
                {
                    genomesToInherit.push_back(sp1->m_scores[0]);
                    continue;
                }

                const auto* sp2 = sp1;
                while (sp1 == sp2)
                {
                    sp2 = &m_generation.m_species[getSpecies(speciesSelector(s_randomGenerator))];
                }

                RandomRealDistribution<float> genomeInSpeciesSelector1(0, sp1->m_adjustedTotalScore);
                RandomRealDistribution<float> genomeInSpeciesSelector2(0, sp2->m_adjustedTotalScore);

                // Select random two genomes
                const Score* g1 = &getGenome(genomeInSpeciesSelector1(s_randomGenerator), sp1->m_scores);
                const Score* g2 = &getGenome(genomeInSpeciesSelector2(s_randomGenerator), sp2->m_scores);

                // Ensure the genome at i1 has a higher fitness
                if (g1->m_fitness < g2->m_fitness)
                {
                    std::swap(g1, g2);
                }

                // Cross over
                newGenomes.push_back(CrossOver(GetGenome(g1->m_index), g1->m_fitness, GetGenome(g2->m_index), g2->m_fitness));
                const float fitness = Evaluate(newGenomes.back());
                if (fitness >= m_generation.m_species[0].m_bestScore.m_fitness)
                {
                    newGenomes.back().m_protect = true;
                }

                genomesToInherit.push_back(Score{ fitness, -1.f, newGenomes.size() - 1 });
            }
        }
    }


    //for (int i = 0; i < (int)m_generation.m_species.size(); ++i)
    //{
    //    if (m_generation.m_species[i].m_scores.size() > 5)
    //    {
    //        assert(m_generation.m_species[i].m_scores[0].m_adjustedFitness > 0);
    //        genomesToInherit.push_back(m_generation.m_species[i].m_scores[0]);
    //    }
    //}

    //if (canCrossOver)
    //{
    //    RandomRealDistribution<float> randomBinary(0, 1.0f);

    //    int numOrgsToCopy = GetNumGenomes() * (1.f - m_config.m_crossOverRate);

    //    // Just copy high score genomes to the next generation
    //    for (int i = (int)genomesToInherit.size(); i < numOrgsToCopy; ++i)
    //    {
    //        genomesToInherit.push_back(getGenome(genomeSelector(s_randomGenerator), m_scores));
    //        assert(genomesToInherit.back().m_adjustedFitness > 0);
    //    }

    //    //std::vector<Genome> newGenomes;
    //    //newGenomes.reserve(numOrgsToCopy - genomesToInherit.size());

    //    // Rest population will be generated by cross over
    //    for (int i = (int)genomesToInherit.size(); i < GetNumGenomes(); ++i)
    //    {
    //        RandomRealDistribution<float> speciesSelector(0, adjustedScoreSum);
    //        auto getSpecies = [this](float f)
    //        {
    //            float currentSum = 0;
    //            for (int i = 0; i < (int)m_generation.m_species.size(); ++i)
    //            {
    //                const auto& species = m_generation.m_species[i];
    //                currentSum += species.m_adjustedTotalScore;
    //                if (currentSum > f)
    //                {
    //                    return i;
    //                }
    //            }

    //            return (int)m_generation.m_species.size() - 1;
    //        };

    //        const Score* g1 = nullptr;
    //        const Score* g2 = nullptr;
    //        if (m_generation.m_species.size() > 1 && randomBinary(s_randomGenerator) < m_config.m_interSpeciesMatingRate)
    //        {
    //            const auto* sp1 = &m_generation.m_species[getSpecies(speciesSelector(s_randomGenerator))];
    //            if (sp1->m_scores.size() == 1)
    //            {
    //                genomesToInherit.push_back(sp1->m_scores[0]);
    //                assert(genomesToInherit.back().m_adjustedFitness > 0);
    //                continue;
    //            }

    //            const auto* sp2 = sp1;
    //            while (sp1 == sp2)
    //            {
    //                sp2 = &m_generation.m_species[getSpecies(speciesSelector(s_randomGenerator))];
    //            }

    //            RandomRealDistribution<float> genomeInSpeciesSelector1(0, sp1->m_adjustedTotalScore);
    //            RandomRealDistribution<float> genomeInSpeciesSelector2(0, sp2->m_adjustedTotalScore);

    //            // Select random two genomes
    //            g1 = &getGenome(genomeInSpeciesSelector1(s_randomGenerator), sp1->m_scores);
    //            g2 = &getGenome(genomeInSpeciesSelector2(s_randomGenerator), sp2->m_scores);
    //        }
    //        else
    //        {
    //            const auto& sp = m_generation.m_species[getSpecies(speciesSelector(s_randomGenerator))];

    //            if (sp.m_scores.size() == 1)
    //            {
    //                genomesToInherit.push_back(sp.m_scores[0]);
    //                assert(genomesToInherit.back().m_adjustedFitness > 0);
    //                continue;
    //            }

    //            RandomRealDistribution<float> genomeInSpeciesSelector(0, sp.m_adjustedTotalScore);

    //            // Select random two genomes
    //            g1 = &getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores);
    //            g2 = &getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores);

    //            while (g1 == g2)
    //            {
    //                g2 = &getGenome(genomeInSpeciesSelector(s_randomGenerator), sp.m_scores);
    //            }
    //        }

    //        // Ensure the genome at i1 has a higher fitness
    //        if (g1->m_fitness < g2->m_fitness)
    //        {
    //            std::swap(g1, g2);
    //        }

    //        // Cross over
    //        newGenomes.push_back(CrossOver(GetGenome(g1->m_index), g1->m_fitness, GetGenome(g2->m_index), g2->m_fitness));

    //        const float fitness = Evaluate(newGenomes.back());
    //        genomesToInherit.push_back(Score{ fitness, -1.f, newGenomes.size() - 1 });
    //    }
    //}
    //else
    //{
    //    // Just randomly select the rest population
    //    for (int i = (int)genomesToInherit.size(); i < GetNumGenomes(); ++i)
    //    {
    //        genomesToInherit.push_back(getGenome(genomeSelector(s_randomGenerator), m_scores));
    //    }
    //}

    std::sort(genomesToInherit.begin(), genomesToInherit.end(), [](const Score& s1, const Score& s2)
    {
        return s1.m_fitness > s2.m_fitness;
    });

    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        int index = genomesToInherit[i].m_index;
        (*m_nextGenGenomesBuffer)[i] = genomesToInherit[i].m_adjustedFitness > 0 ? AccessGenome(index) : newGenomes[index];
    }

    {
        if (Evaluate((*m_generation.m_genomes)[0]) > Evaluate((*m_nextGenGenomesBuffer)[0]) &&
            m_generation.m_species[0].m_scores.size() >= 5)
        {
            // Fitness regressed!
            int a = 0;
            a = 1;
        }
    }

    GenomeList tmp = m_generation.m_genomes;
    m_generation.m_genomes = m_nextGenGenomesBuffer;
    m_nextGenGenomesBuffer = tmp;

    m_generation.m_generationId++;
}

void NEAT::Mutate()
{
    // Mutation weights
    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        Genome& genome = AccessGenome(i);

        if (genome.m_protect) continue;

        for (auto& elem : genome.m_connectionGenes)
        {
            // Invoke mutation at random rate
            if (GetRandomProbability() < m_config.m_weightMutationRate)
            {
                auto& connection = elem.second;

                // Only perturb weight at random rate
                if (GetRandomProbability() < m_config.m_weightPerturbationRate)
                {
                    RandomIntDistribution<int> randomBinary(0, 1);
                    float sign = randomBinary(s_randomGenerator) > 0 ? 1.0f : -1.0f;
                    connection.m_weight += sign * 0.05f;
                    if (connection.m_weight > 1) connection.m_weight = 1.0f;
                    if (connection.m_weight < -1) connection.m_weight = -1.f;
                }

                // Assign a completely new weight
                if (GetRandomProbability() < m_config.m_weightNewValueRate)
                {
                    connection.m_weight = GetRandomWeight();
                }
            }
        }
    }

    NewlyAddedNodes newNodes;

    // Apply topological mutations
    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        Genome& genome = AccessGenome(i);
        if (genome.m_protect)
        {
            continue;
        }

        // Add a new node at random rate
        if (GetRandomProbability() < m_config.m_nodeAdditionRate)
        {
            NodeAddedInfo node = AddNewNode(genome);
            if (node.m_newNode != s_invalidNodeGeneId)
            {
                newNodes[node.m_oldConnection].push_back(node);
            }
        }
    }

    EnsureUniqueGeneIndices(newNodes);

    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        const Genome& g = GetGenome(i);
        CheckSanity(g);
    }

    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        Genome& genome = AccessGenome(i);
        if (genome.m_protect)
        {
            genome.m_protect = false;
            continue;
        }

        // Add a new connection at random rate
        if (GetRandomProbability() < m_config.m_connectionAdditionRate)
        {
            AddNewConnection(genome, m_config.m_allowCyclicNetwork);
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

        NodeGeneId inNode = info.m_genome.m_connectionGenes[info.m_oldConnection].m_inNode;
        NodeGeneId outNode = info.m_genome.m_connectionGenes[info.m_oldConnection].m_outNode;

        for (size_t i = 1; i < genomes.size(); ++i)
        {
            const auto thisInfo = genomes[i];
            Genome& genome = genomes[i].m_genome;

            assert(genome.m_nodeLinks.find(info.m_newNode) == genome.m_nodeLinks.end());
            genome.m_nodeLinks[info.m_newNode] = genome.m_nodeLinks[thisInfo.m_newNode];
            genome.m_nodeLinks.erase(thisInfo.m_newNode);

            assert(genome.m_nodeLinks[info.m_newNode].m_incomings.size() == 1);
            assert(genome.m_nodeLinks[info.m_newNode].m_outgoings.size() == 1);
            genome.m_nodeLinks[info.m_newNode].m_incomings[0] = info.m_newConnection1;
            genome.m_nodeLinks[info.m_newNode].m_outgoings[0] = info.m_newConnection2;

            for (auto& innovId : genome.m_nodeLinks[outNode].m_incomings)
            {
                if (innovId == thisInfo.m_newConnection2)
                {
                    innovId = info.m_newConnection2;
                    break;
                }
            }
            for (auto& innovId : genome.m_nodeLinks[inNode].m_outgoings)
            {
                if (innovId == thisInfo.m_newConnection1)
                {
                    innovId = info.m_newConnection1;
                    break;
                }
            }

            assert(genome.m_connectionGenes.find(info.m_newConnection1) == genome.m_connectionGenes.end());
            genome.m_connectionGenes[info.m_newConnection1] = genome.m_connectionGenes[thisInfo.m_newConnection1];
            genome.m_connectionGenes[info.m_newConnection1].m_innovId = info.m_newConnection1;
            genome.m_connectionGenes[info.m_newConnection1].m_outNode = info.m_newNode;
            genome.m_connectionGenes.erase(thisInfo.m_newConnection1);

            assert(genome.m_connectionGenes.find(info.m_newConnection2) == genome.m_connectionGenes.end());
            genome.m_connectionGenes[info.m_newConnection2] = genome.m_connectionGenes[thisInfo.m_newConnection2];
            genome.m_connectionGenes[info.m_newConnection2].m_innovId = info.m_newConnection2;
            genome.m_connectionGenes[info.m_newConnection2].m_inNode = info.m_newNode;
            genome.m_connectionGenes.erase(thisInfo.m_newConnection2);

            m_innovationHistory.erase(NodePair{ inNode, thisInfo.m_newNode });
            m_innovationHistory.erase(NodePair{ thisInfo.m_newNode, outNode });

            assert(CheckSanity(genome));
        }
    }
}

void NEAT::Speciation()
{
    for (auto& species : m_generation.m_species)
    {
        species.m_scores.clear();
        if (species.m_stagnantGenerationCount == 0)
        {
            species.m_previousBestFitness = species.m_bestScore.m_fitness;
        }
        species.m_adjustedTotalScore = 0.f;
        species.m_bestScore.m_fitness = 0.f;
        species.m_bestScore.m_adjustedFitness = 0.f;
    }

    for (size_t i = 0; i < m_config.m_numOrganismsInGeneration; ++i)
    {
        Genome& genome = AccessGenome(i);
        genome.m_species = s_invalidSpeciesId;
        for (auto& sp : m_generation.m_species)
        {
            const auto& representative = sp.m_representative;
            if (representative.m_connectionGenes.size() > 0)
            {
                if (CalculateDistance(genome, representative) < m_config.m_speciationDistThreshold)
                {
                    genome.m_species = sp.m_id;
                    sp.m_scores.push_back(m_scores[i]);

                    if (sp.m_bestScore.m_fitness < m_scores[i].m_fitness)
                    {
                        sp.m_bestScore = m_scores[i];
                    }
                    break;
                }
            }
        }

        if (genome.m_species == s_invalidSpeciesId)
        {
            genome.m_species = m_currentSpeciesId;
            std::vector<Species>& sp = m_generation.m_species;
            sp.push_back(Species{ m_currentSpeciesId++ });
            sp.back().m_scores.push_back(m_scores[i]);
            sp.back().m_bestScore = m_scores[i];
            sp.back().m_representative = genome;
        }
    }

    for (auto itr = m_generation.m_species.begin(); itr != m_generation.m_species.end();)
    {
        auto& species = *itr;
        if (species.m_scores.size() == 0)
        {
            itr = m_generation.m_species.erase(itr);
            continue;
        }

        RandomIntDistribution<uint32_t> randomInt(0, species.m_scores.size() - 1);
        int representativeIndex = species.m_scores[randomInt(s_randomGenerator)].m_index;
        species.m_representative = GetGenome(representativeIndex);

        bool extinct = false;
        if (species.m_previousBestFitness >= species.m_bestScore.m_fitness)
        {
            ++species.m_stagnantGenerationCount;
            if (species.m_stagnantGenerationCount >= 15 && m_generation.m_species.size() > 2)
            {
                extinct = true;
            }
        }
        else
        {
            species.m_stagnantGenerationCount = 0;
        }

        if (!extinct || !m_config.m_extinctStagnantSpecies)
        {
            float denom = 1.0f / (float)species.m_scores.size();
            for (auto& s : species.m_scores)
            {
                s.m_adjustedFitness = s.m_fitness * denom;
                m_scores[s.m_index].m_adjustedFitness = s.m_adjustedFitness;
                species.m_adjustedTotalScore += s.m_adjustedFitness;
            }

            if (species.ShouldProtectBest())
            {
                AccessGenome(species.m_bestScore.m_index).m_protect = true;
            }

            // Sort genomes stored in the species by fitness
            // This is necessary to remove lowest fitness genomes during selection
            std::sort(species.m_scores.begin(), species.m_scores.end(), [](const Score& s1, const Score& s2)
            {
                return s1.m_fitness > s2.m_fitness;
            });
        }
        else
        {
            for (auto& s : species.m_scores)
            {
                m_scores[s.m_index].m_fitness = 0.f;
                m_scores[s.m_index].m_adjustedFitness = 0.f;
                s.m_fitness = 0.f;
            }
            species.m_adjustedTotalScore = 0.f;
            itr = m_generation.m_species.erase(itr);
            continue;
        }

        ++itr;
    }
}

// Calculate distance between two genomes based on their topologies and weights
float NEAT::CalculateDistance(const Genome& genome1, const Genome& genome2) const
{
    const auto& cGenes1 = genome1.m_connectionGenes;
    const auto& cGenes2 = genome2.m_connectionGenes;
    const size_t numConnectionsP1 = cGenes1.size();
    const size_t numConnectionsP2 = cGenes2.size();
    const size_t numConnectionsLarge = numConnectionsP1 > numConnectionsP2 ? numConnectionsP1 : numConnectionsP2;
    const size_t numConnectionsSmall = numConnectionsP1 > numConnectionsP2 ? numConnectionsP2 : numConnectionsP1;
    size_t numMismatches = 0;
    size_t numMatches = 0;
    float weightDifference = 0.f;

    auto itr1 = cGenes1.begin();
    auto itr2 = cGenes2.begin();

    while (itr1 != cGenes1.end() && itr2 != cGenes2.end())
    {
        const ConnectionGene& cGene1 = itr1->second;
        const ConnectionGene& cGene2 = itr2->second;
        if (cGene1.m_innovId == cGene2.m_innovId)
        {
            weightDifference += std::fabs((/*cGene1.m_enabled ? */cGene1.m_weight/* : 0.f*/) - (/*cGene2.m_enabled ? */cGene2.m_weight/* : 0.f*/));
            ++numMatches;
            ++itr1;
            ++itr2;
        }
        else
        {
            ++numMismatches;
            if (cGene1.m_innovId < cGene2.m_innovId)
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
        int n = numConnectionsSmall >= 20 ? numConnectionsLarge : 1;
        return (float)numMismatches / (float)n + 0.4f * (weightDifference / (float)numMatches);
    }
}

// Evaluate a genome and return its fitness
float NEAT::Evaluate(const Genome& genom) const
{
    if (m_fitnessFunc)
    {
        return m_fitnessFunc(genom);
    }

    return 0.0f;
}

// Evaluate value of each node recursively
void NEAT::EvaluateRecursive(const Genome& genome, NodeGeneId nodeId, std::vector<NodeGeneId>& evaluatingNodes, std::unordered_map<NodeGeneId, float>& values) const
{
    assert(m_config.m_allowCyclicNetwork == false);

    float val = 0.f;

    if (values.find(nodeId) != values.end())
    {
        return;
    }

    for (auto innovId : genome.m_nodeLinks.at(nodeId).m_incomings)
    {
        const auto connectionGene = GetConnectionGene(genome, innovId);
        assert(connectionGene != nullptr);

        if (!connectionGene->m_enabled) continue;

        auto incomingNodeId = connectionGene->m_inNode;
        bool alreadyEvaluating = false;

        if (values.find(incomingNodeId) == values.end())
        {
            if (m_config.m_allowCyclicNetwork)
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

            if (m_config.m_allowCyclicNetwork)
            {
                evaluatingNodes.resize(evaluatingNodes.size() - 1);
            }
        }

        val += values[incomingNodeId] * connectionGene->m_weight;
    }

    values[nodeId] = m_activationFuncs[m_generation.m_nodeGenes[nodeId].m_activationFuncId](val);
}

// Set up node used for the initial network
void NEAT::SetupInitialNodeGenes()
{
    CreateNewNode(NodeGeneType::Input);
    CreateNewNode(NodeGeneType::Input);
    CreateNewNode(NodeGeneType::Output);
}

// Create default genome for the initial generation
auto NEAT::CreateDefaultInitialGenome() -> Genome
{
    // Create two input nodes, one output node and two connections with random weight

    Genome genomeOut;

    CreateNewNode(NodeGeneType::Input);

    NodeGeneId input1 = 0;
    NodeGeneId input2 = 1;
    NodeGeneId output = 2;
    genomeOut.m_nodeLinks[input1] = Genome::Links();
    genomeOut.m_nodeLinks[input2] = Genome::Links();
    genomeOut.m_nodeLinks[output] = Genome::Links();

    RandomRealDistribution<float> randomf(-1.f, 1.f);
    Connect(genomeOut, input1, output, randomf(s_randomGenerator));
    Connect(genomeOut, input2, output, randomf(s_randomGenerator));

    return genomeOut;
}

bool NEAT::CheckSanity(const Genome& genome) const
{
#ifdef _DEBUG
    if (!m_config.m_enableSanityCheck)
    {
        return true;
    }

    // Check if there's no duplicated connection genes for the same in and out nodes
    {
        const auto& conns = genome.m_connectionGenes;
        for (auto itr1 = conns.begin(); itr1 != conns.end(); ++itr1)
        {
            auto itr2 = itr1;
            for (++itr2; itr2 != conns.end(); ++itr2)
            {
                const auto& c1 = itr1->second;
                const auto& c2 = itr2->second;
                if (c1.m_inNode == c2.m_inNode && c1.m_outNode == c2.m_outNode)
                {
                    return false;
                }
            }
        }
    }

    {
        for (auto elem : genome.m_nodeLinks)
        {
            const Genome::Links& links = elem.second;
            if (links.m_numEnabledIncomings > (int)links.m_incomings.size() ||
                links.m_numEnabledOutgoings > (int)links.m_outgoings.size())
            {
                return false;
            }

            int count = 0;
            for (InnovationId iid : links.m_incomings)
            {
                if (genome.m_connectionGenes.at(iid).m_enabled)
                {
                    ++count;
                }
            }
            if (count != links.m_numEnabledIncomings)
            {
                return false;
            }

            count = 0;
            for (InnovationId iid : links.m_outgoings)
            {
                if (genome.m_connectionGenes.at(iid).m_enabled)
                {
                    ++count;
                }
            }
            if (count != links.m_numEnabledOutgoings)
            {
                return false;
            }
        }
    }

    {
        for (auto elem : genome.m_nodeLinks)
        {
            if (m_generation.m_nodeGenes[elem.first].m_type == NodeGeneType::Hidden)
            {
                const Genome::Links& links = elem.second;
                if (links.m_incomings.size() == 0 || links.m_outgoings.size() == 0)
                {
                    return false;
                }
            }
        }
    }

    if (!m_config.m_allowCyclicNetwork)
    {
        for (const auto con : genome.m_connectionGenes)
        {
            if (con.second.m_enabled && !CanAddConnectionWithoutCyclic(genome, con.second.m_inNode, con.second.m_outNode))
            {
                return false;
            }
        }
    }

#endif
    return true;
}


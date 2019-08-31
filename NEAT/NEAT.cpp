#include "NEAT.h"

#include <bitset>
#include <algorithm>
#include <stack>
#include <fstream>
#include <iostream>

std::default_random_engine NEAT::s_randomGenerator;

#define TEST_PREVENT_TO_ADD_NEW_CONNECTION_TO_DEADEND_HIDDEN_NODE 1
#define TEST_PROTECT_NEWLY_CREATED_GENOME_BY_CROSS_OVER 1

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
    if (m_config.m_numGenomesInGeneration == 0)
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
    for (int i = 0; i < m_config.m_numGenomesInGeneration; ++i)
    {
        AccessGenome(i) = CreateDefaultInitialGenome(false);
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
        buffer->resize(m_config.m_numGenomesInGeneration);
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

    m_innovationHistory.clear();

    // Reset global ids
    m_currentInnovationId = 0;
    m_currentSpeciesId = 0;

    // Allocate memory for scores
    m_scores.resize(m_config.m_numGenomesInGeneration);

    m_evaluationCount = 0;
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
    float sumFitness = 0;
    float maxFitness = 0;
    int bestGenomeId = -1;

    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        const Genome& genome = GetGenome(i);
        const float fitness = Evaluate(genome);

        sumFitness += fitness;
        if (fitness > maxFitness)
        {
            maxFitness = fitness;
            bestGenomeId = i;
        }
    }

    const int bestSpeciesId = GetGenome(0).m_species;
    const Species* bestSpecies = nullptr;
    for (const auto& sp : m_generation.m_species)
    {
        if (sp->m_id == bestSpeciesId)
        {
            bestSpecies = sp;
            break;
        }
    }

    std::cout << "======================" << std::endl;
    std::cout << "Generation " << m_generation.m_generationId << std::endl;
    std::cout << "Average fitness among " << GetNumGenomes() << " genomes : " << sumFitness / (float)GetNumGenomes() << std::endl;
    std::cout << "Maximum fitness: Species " << bestSpeciesId << ": Genome " << bestGenomeId << " - Score: " << maxFitness << std::endl;
    std::cout << "Number of species: " << m_generation.m_species.size() << std::endl;
    if (bestSpecies)
    {
        std::cout << "Number of genomes in the best species: " << bestSpecies->m_scores.size() << std::endl;
    }
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
                const Species* spc = m_generation.m_species[i];
                printTabs(); json << "{" << endl;
                {
                    Scope s(tabLevel);
                    printTabs(); json << "\"Id\" : " << spc->m_id << "," << endl;
                    printTabs(); json << "\"BestScore\" : " << spc->m_bestScore.m_fitness << "," << endl;
                    printTabs(); json << "\"BestScoreGenomeIndex\" : " << spc->m_bestScore.m_index << endl;
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
                        for (auto& elem : genome.m_nodes)
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
                            if (i != (int)genome.m_nodes.size())
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
        RandomIntDistribution<uint32_t> distribution(0, (int)m_activationFuncs.size());
        node.m_activationFuncId = distribution(s_randomGenerator);
    }

    NodeGeneId nodeGeneId = (int)m_generation.m_nodeGenes.size();
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

// Add a new connection between random two nodes
void NEAT::AddNewConnection(Genome& genome)
{
    const size_t numNodes = genome.GetNumNodes();

    // Collect all node genes where we can add a new connection gene first
    std::vector<NodeGeneId> outNodeCandidates;
    {
        for (const auto& elem : genome.m_nodes)
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
                if (genome.GetIncommingConnections(nodeId).size() < numNodes)
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
            std::set<NodeGeneId> alreadyConnectedNodes;

            // Mark nodes connected to outNode
            for (auto innovId : genome.GetIncommingConnections(outNodeId))
            {
                const ConnectionGene* cgene = GetConnectionGene(genome, innovId);
                assert(cgene);
                alreadyConnectedNodes.insert(cgene->m_inNode);
            }

            // Find all nodes not connected to outNode
            for (const auto& elem : genome.m_nodes)
            {
                const NodeGeneId nodeId = elem.first;
                const NodeGene& node = GetNodeGene(nodeId);

                if (!m_config.m_allowRecurrentNetwork && nodeId == outNodeId) continue;

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
                    if (alreadyConnectedNodes.find(nodeId) == alreadyConnectedNodes.end())
                    {
                        // Check recurrent network
                        if (m_config.m_allowRecurrentNetwork || CanAddConnectionWithoutRecurrent(genome, nodeId, outNodeId))
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

// Remove one connection from genome
// If the removal leads a connected node isolated (connected to nothing), remove such node as well
void NEAT::RemoveConnection(Genome& genome) const
{
    // Create an array of connections
    std::vector<InnovationId> connections;
    {
        connections.reserve(genome.GetNumConnections());
        for (const auto& elem : genome.m_connectionGenes)
        {
            connections.push_back(elem.first);
        }
    }

    // Select a random connection
    if (connections.size() > 0)
    {
        const InnovationId innovId = SelectRandomConnectionGene(connections);
        RemoveConnection(genome, innovId);
    }
}

void NEAT::RemoveConnection(Genome& genome, InnovationId innovId) const
{
    const ConnectionGene* connection = GetConnectionGene(genome, innovId);
    assert(connection);
    const NodeGeneId inNode = connection->m_inNode;
    const NodeGeneId outNode = connection->m_outNode;
    const bool enabled = connection->m_enabled;

    // Remove the connection
    genome.m_connectionGenes.erase(innovId);

    // Update node links
    {
        Genome::Node& links = genome.m_nodes[outNode];
        std::vector<InnovationId>& incomings = links.m_incomings;
        for (auto itr = incomings.begin(); itr != incomings.end(); ++itr)
        {
            if (*itr == innovId)
            {
                incomings.erase(itr);
                if (enabled) --links.m_numEnabledIncomings;
                break;
            }
        }
    }
    {
        Genome::Node& links = genome.m_nodes[inNode];
        std::vector<InnovationId>& outgoings = links.m_outgoings;
        for (auto itr = outgoings.begin(); itr != outgoings.end(); ++itr)
        {
            if (*itr == innovId)
            {
                outgoings.erase(itr);
                if (enabled) --links.m_numEnabledOutgoings;
                break;
            }
        }
    }

    // If the connected node became completely isolated, delete them too
    auto RemoveNode = [&genome, this](NodeGeneId node)
    {
        if (GetNodeGene(node).m_type == NodeGeneType::Hidden)
        {
            if (genome.GetIncommingConnections(node).size() == 0 && genome.GetOutgoingConnections(node).size() == 0)
            {
                genome.m_nodes.erase(node);
            }
        }
    };

    RemoveNode(inNode);
    RemoveNode(outNode);

    assert(CheckSanity(genome));
}

// Return false if adding a connection between srcNode to targetNode makes the network recurrent
bool NEAT::CanAddConnectionWithoutRecurrent(const Genome& genome, NodeGeneId srcNode, NodeGeneId targetNode) const
{
    if (srcNode == targetNode)
    {
        return false;
    }

    // TODO: This bit array could be way bigger than necessary. Replace it with a better solution.
    std::vector<int> flag;
    flag.resize((int)m_generation.m_nodeGenes.size() / sizeof(int) + 1, 0);

    // Test if we can reach to the targetNode from srcNode by following connections reversely
    std::stack<NodeGeneId> stack;
    stack.push(srcNode);
    while (!stack.empty())
    {
        const NodeGeneId node = stack.top();
        stack.pop();

        // Test all incoming connections of this node
        for (const InnovationId innovId : genome.GetIncommingConnections(node))
        {
            const ConnectionGene* con = GetConnectionGene(genome, innovId);
            assert(con);

            // Ignore disabled node
            if (!con->m_enabled) continue;

            const NodeGeneId inNode = con->m_inNode;

            if (inNode == targetNode)
            {
                // Reached to the target node
                // The new connection will make the network recurrent
                return false;
            }

            const int index = inNode / sizeof(int);
            const int offset = inNode % sizeof(int);
            // Add this node to the stack if we haven't yet
            if (((flag[index] >> offset) & 1) == 0)
            {
                stack.push(inNode);
                flag[index] |= 1 << offset;
            }
        }
    }

    return true;
}

// Implementation of generating a new generation
void NEAT::GenerateNewGeneration(bool printFitness)
{
    // First mutate genomes
    Mutate();

    // Second apply diversity protection treatment
    if (m_config.m_diversityProtection == DiversityProtectionMethod::Speciation)
    {
        // Perform speciation
        Speciation();
    }
    else if (m_config.m_diversityProtection == DiversityProtectionMethod::MorphologicalInnovationProtection)
    {
        // Not implemented yet
        assert(0);
    }

    // Third select genomes to the next generation
    SelectGenomes();

    if (printFitness)
    {
        PrintFitness();
    }
}

// Apply mutation
void NEAT::Mutate()
{
    // Mutate weights
    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        Genome& genome = AccessGenome(i);

        if (genome.m_protect) continue;

        // For each connection
        for (auto& elem : genome.m_connectionGenes)
        {
            // Invoke mutation at random rate
            if (GetRandomProbability() < m_config.m_weightMutationRate)
            {
                auto& connection = elem.second;

                // Perturb weight
                if (GetRandomProbability() < m_config.m_weightPerturbationRate)
                {
                    RandomIntDistribution<int> randomBinary(0, 1);
                    float sign = randomBinary(s_randomGenerator) > 0 ? 1.0f : -1.0f;
                    connection.m_weight += sign * m_config.m_weightPerturbation;

                    // Cramp weight
                    if (connection.m_weight > m_config.m_maximumWeight) connection.m_weight = m_config.m_maximumWeight;
                    if (connection.m_weight < m_config.m_minimumWeight) connection.m_weight = m_config.m_minimumWeight;
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

    // Make sure that the same topological changes have the same id
    EnsureUniqueGeneIndices(newNodes);

    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        Genome& genome = AccessGenome(i);

        if (genome.m_protect)
        {
            continue;
        }

        // Add a new connection at random rate
        if (GetRandomProbability() < m_config.m_connectionAdditionRate)
        {
            AddNewConnection(genome);
        }

        if (m_config.m_removeConnectionsByMutation)
        {
            if (GetRandomProbability() < m_config.m_connectionRemovalRate)
            {
                RemoveConnection(genome);
            }
        }
    }
}

// Make sure that the same topological changes have the same id
void NEAT::EnsureUniqueGeneIndices(const NewlyAddedNodes& newNodes)
{
    // Check duplicated newly added node genes
    for (const auto& elem : newNodes)
    {
        const auto& genomes = elem.second;

        // Use information of genome added first
        const NodeAddedInfo& info = genomes[0];

        // in node and out node where the new node was added
        NodeGeneId inNode  = info.m_genome.m_connectionGenes[info.m_oldConnection].m_inNode;
        NodeGeneId outNode = info.m_genome.m_connectionGenes[info.m_oldConnection].m_outNode;

        // Update other genomes
        for (size_t i = 1; i < genomes.size(); ++i)
        {
            // Genome to update
            Genome& genome = genomes[i].m_genome;
            const NodeAddedInfo& thisInfo = genomes[i];

            // Update node links
            assert(!genome.HasNode(info.m_newNode));
            Genome::Node& node = genome.m_nodes[info.m_newNode];
            node = genome.m_nodes[thisInfo.m_newNode];
            assert(genome.GetIncommingConnections(info.m_newNode).size() == 1);
            assert(genome.GetOutgoingConnections(info.m_newNode).size() == 1);
            node.m_incomings[0] = info.m_newConnection1;
            node.m_outgoings[0] = info.m_newConnection2;
            genome.m_nodes.erase(thisInfo.m_newNode);

            for (auto& innovId : genome.m_nodes[outNode].m_incomings)
            {
                if (innovId == thisInfo.m_newConnection2)
                {
                    innovId = info.m_newConnection2;
                    break;
                }
            }
            for (auto& innovId : genome.m_nodes[inNode].m_outgoings)
            {
                if (innovId == thisInfo.m_newConnection1)
                {
                    innovId = info.m_newConnection1;
                    break;
                }
            }

            // Update connections
            assert(!genome.HasConnection(info.m_newConnection1));
            ConnectionGene& c1 = genome.m_connectionGenes[info.m_newConnection1];
            c1 = genome.m_connectionGenes[thisInfo.m_newConnection1];
            c1.m_innovId = info.m_newConnection1;
            c1.m_outNode = info.m_newNode;
            genome.m_connectionGenes.erase(thisInfo.m_newConnection1);

            assert(!genome.HasConnection(info.m_newConnection2));
            ConnectionGene& c2 = genome.m_connectionGenes[info.m_newConnection2];
            c2 = genome.m_connectionGenes[thisInfo.m_newConnection2];
            c2.m_innovId = info.m_newConnection2;
            c2.m_inNode = info.m_newNode;
            genome.m_connectionGenes.erase(thisInfo.m_newConnection2);

            // Fix innovation history
            m_innovationHistory.erase(NodePair{ inNode, thisInfo.m_newNode });
            m_innovationHistory.erase(NodePair{ thisInfo.m_newNode, outNode });

            assert(CheckSanity(genome));
        }
    }
}

// Perform speciation
void NEAT::Speciation()
{
    // Clear information of the previous generation
    for (auto& species : m_generation.m_species)
    {
        // Update the best fitness
        if (species->m_stagnantGenerationCount == 0)
        {
            species->m_previousBestFitness = species->GetBestFitness();
        }

        species->m_scores.clear();
        species->m_adjustedTotalScore = 0.f;
        species->m_adjustedScoreEliminatedLows = 0.f;
        species->m_bestScore.m_fitness = 0.f;
        species->m_bestScore.m_adjustedFitness = 0.f;
        species->m_bestScore.m_index = -1;
    }

    // Assign each genome to an existing species which has close topology, or create a new one if there's none
    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        Genome& genome = AccessGenome(i);

        // Reset protect flag
        genome.m_protect = false;

        // Reset species id
        genome.m_species = s_invalidSpeciesId;

        // Evaluate the genome and store its score
        Score& score = m_scores[i];
        {
            score.m_index = i;
            score.m_adjustedFitness = 0.f;
            score.m_fitness = Evaluate(genome);
        }

        // Check distance against each species
        for (auto& sp : m_generation.m_species)
        {
            const auto& representative = sp->m_representative;
            if (CalculateDistance(genome, representative) < m_config.m_speciationDistThreshold)
            {
                // Found a species for this genome
                genome.m_species = sp->m_id;
                sp->m_scores.push_back(score);

                if (sp->GetBestFitness() < score.m_fitness)
                {
                    sp->m_bestScore = score;
                }
                break;
            }
        }

        // No species close enough was found, create a new one
        if (genome.m_species == s_invalidSpeciesId)
        {
            std::vector<Species*>& spList = m_generation.m_species;
            spList.push_back(new Species{ m_currentSpeciesId++ });
            Species* newSpecies = spList.back();
            newSpecies->m_scores.push_back(score);
            newSpecies->m_bestScore = score;
            newSpecies->m_representative = genome;
            genome.m_species = newSpecies->m_id;
        }
    }

    int numSpeciesToDelete = 0;
    for (Species* species : m_generation.m_species)
    {
        // Delete species which no genomes fell into
        if (species->m_scores.size() == 0)
        {
            species->m_bestScore.m_fitness = 0.f;
            ++numSpeciesToDelete;
            continue;
        }

        // Select a random representative of this species
        {
            RandomIntDistribution<uint32_t> randomInt(0, (int)species->m_scores.size() - 1);
            int representativeIndex = species->m_scores[randomInt(s_randomGenerator)].m_index;
            species->m_representative = GetGenome(representativeIndex);
        }

        // Check extinction
        if (m_config.m_extinctStagnantSpecies)
        {
            // Check if this species has made any progress
            if (species->m_previousBestFitness >= species->GetBestFitness())
            {
                // Increment stagnant generations
                ++species->m_stagnantGenerationCount;

                if (species->m_stagnantGenerationCount >= m_config.m_numGenerationsToExtinctSpecies &&
                    GetNumSpecies() > 2)
                {
                    // Extinct this species
                    species->m_bestScore.m_fitness = 0.f;
                    ++numSpeciesToDelete;
                    continue;
                }
            }
            else
            {
                // Reset stagnant count
                species->m_stagnantGenerationCount = 0;
            }
        }

        // Calculate adjusted fitness
        const float denom = 1.0f / (float)species->GetNumGenomes();
        for (auto& s : species->m_scores)
        {
            s.m_adjustedFitness = s.m_fitness * denom;
            m_scores[s.m_index].m_adjustedFitness = s.m_adjustedFitness;
            species->m_adjustedTotalScore += s.m_adjustedFitness;
        }

        // Mark the best genome as protected
        if (species->ShouldProtectBest())
        {
            AccessGenome(species->m_bestScore.m_index).m_protect = true;
        }

        // Sort genomes stored in the species by fitness in descending order
        // This is necessary to remove lowest fitness genomes during selection
        std::sort(species->m_scores.rbegin(), species->m_scores.rend());
    }

    // Sort species in descending order
    std::sort(m_generation.m_species.begin(), m_generation.m_species.end(), [](const Species* s1, const Species* s2)
    {
        return s2->m_bestScore < s1->m_bestScore;
    });

    // If the entire population hasn't been progressing for a certain number of generations,
    // extinct all species except the top two
    if (m_config.m_extinctWholeGeneration &&
        GetNumSpecies() > 2 &&
        m_generation.m_prevBestFitness >= m_generation.m_species[0]->GetBestFitness())
    {
        ++m_generation.m_stagnantGenerationCount;
        if (m_generation.m_stagnantGenerationCount >= m_config.m_numGenerationsToExtinctMostSpecies)
        {
            // Find the number of species which have the best score and the second best score in the generation
            int numSpKeep = 1;
            {
                float bestScore = m_generation.m_species[0]->GetBestFitness();
                float secondScore = 0.f;
                for (int i = 1; i < GetNumSpecies(); ++i)
                {
                    if (bestScore == m_generation.m_species[i]->GetBestFitness() ||
                        secondScore == m_generation.m_species[i]->GetBestFitness())
                    {
                        ++numSpKeep;
                    }
                    else if (m_generation.m_species[i]->GetBestFitness() > secondScore)
                    {
                        secondScore = m_generation.m_species[i]->GetBestFitness();
                        ++numSpKeep;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            numSpeciesToDelete = GetNumSpecies() - numSpKeep;
            m_generation.m_stagnantGenerationCount = 0;
        }
    }
    else
    {
        m_generation.m_prevBestFitness = m_generation.m_species[0]->GetBestFitness();
        m_generation.m_stagnantGenerationCount = 0;
    }

    // Delete species
    // Because species are now sorted by best fitness, we just need to delete the last N elements in the array
    if (GetNumSpecies() > numSpeciesToDelete)
    {
        for (int i = 0; i < numSpeciesToDelete; ++i)
        {
            const int index = GetNumSpecies() - 1 - i;
            assert(index > 0);
            Species* species = m_generation.m_species[index];

            // Clear all genomes in this species
            for (auto& s : species->m_scores)
            {
                m_scores[s.m_index].m_fitness = 0.f;
                m_scores[s.m_index].m_adjustedFitness = 0.f;
            }

            delete species;
        }
        const int newSize = GetNumSpecies() - numSpeciesToDelete;
        assert(newSize > 0);
        m_generation.m_species.resize(newSize);
    }
}

// Calculate distance between two genomes based on their topologies and weights
float NEAT::CalculateDistance(const Genome& genome1, const Genome& genome2) const
{
    const int numConsP1 = genome1.GetNumConnections();
    const int numConsP2 = genome2.GetNumConnections();

    if (numConsP1 == 0 && numConsP2 == 0)
    {
        return 0;
    }

    const int numConsLarge = numConsP1 > numConsP2 ? numConsP1 : numConsP2;
    const int numConsSmall = numConsP1 > numConsP2 ? numConsP2 : numConsP1;
    int numMismatches = 0;
    int numMatches = 0;
    float weightDifference = 0.f;

    // Count matching and mismatching connections
    // and accumulate difference of weights for matching genes
    const auto& cGenes1 = genome1.m_connectionGenes;
    const auto& cGenes2 = genome2.m_connectionGenes;
    auto itr1 = cGenes1.begin();
    auto itr2 = cGenes2.begin();
    while (itr1 != cGenes1.end() && itr2 != cGenes2.end())
    {
        const ConnectionGene& cGene1 = itr1->second;
        const ConnectionGene& cGene2 = itr2->second;
        if (cGene1.m_innovId == cGene2.m_innovId)
        {
            // [TODO]: Should we treat weight of disabled gene as zero?
            weightDifference += std::fabs(cGene1.m_weight - cGene2.m_weight);
            ++numMatches;
            ++itr1;
            ++itr2;
        }
        else
        {
            ++numMismatches;
            if (cGene1.m_innovId < cGene2.m_innovId)
                ++itr1;
            else
                ++itr2;
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
        numMatches = 1;
    }

    return (float)numMismatches / (float)(numConsSmall >= 20 ? numConsLarge : 1) +
        m_config.m_weightScaleForDistance * (weightDifference / (float)numMatches);
}

// Select genomes to the next generation
void NEAT::SelectGenomes()
{
    // List of genomes and their scores to inherit to the next generation
    std::vector<Score> genomesToInherit;
    genomesToInherit.reserve(GetNumGenomes());

    // Compute sum of adjusted scores of all genomes
    float adjustedScoreSum = 0;
    for (const auto s : m_generation.m_species)
    {
        adjustedScoreSum += s->m_adjustedTotalScore;
    }

    // Temporary buffer to store genomes newly created by cross over
    std::vector<Genome> newGenomes;

    // Distribute population over the species and inherit genomes from each species
    {
        const int totalPopulationForSpecies = int(GetNumGenomes() * (1.0f - m_config.m_interSpeciesMatingRate));
        const float invAdjustedScoreSum = 1.0f / adjustedScoreSum;
        for (Species* sp : m_generation.m_species)
        {
            const int numGenomes = sp->GetNumGenomes();

            // Remove lowest fitness genomes from selection
            // and adjust total score of this species by subtracting scores from those low genomes
            const int numGenomesToRemove = (int)(sp->m_scores.size() * m_config.m_lowerGenomeEliminationRate);
            sp->m_adjustedScoreEliminatedLows = sp->m_adjustedTotalScore;
            for (int i = numGenomes - numGenomesToRemove; i < numGenomes; ++i)
            {
                sp->m_adjustedScoreEliminatedLows -= sp->m_scores[i].m_adjustedFitness;
            }

            // Calculate allocated population for this species
            const int spPopulation = int(totalPopulationForSpecies * sp->m_adjustedTotalScore * invAdjustedScoreSum);

            // Counter of population of this species
            int population = 0;

            // Guarantee that the best genomes is copied to the next gen if this species is big enough, 
            if (sp->ShouldProtectBest())
            {
                genomesToInherit.push_back(sp->m_scores[0]);
                ++population;
            }

            // Copy some genomes to the next gen
            {
                const int numGenomesToCopy = int(spPopulation * (1.f - m_config.m_crossOverRate));
                for (; population < numGenomesToCopy; ++population)
                {
                    genomesToInherit.push_back(SelectGenome(GetRandomValue(sp->m_adjustedScoreEliminatedLows), sp->m_scores));
                }
            }

            // The rest of population is generated by cross over
            while (population < spPopulation)
            {
                genomesToInherit.push_back(GetInheritanceFromSpecies(*sp, newGenomes));
                ++population;
            }
        }

        assert((int)genomesToInherit.size() <= GetNumGenomes());

        // It's possible that entire population hasn't reached to totalPopulationForSpecies yet
        // due to rounding population of each species from float to int.
        // For remaining population, we just select random species and perform cross over.
        for (int i = (int)genomesToInherit.size(); i < totalPopulationForSpecies; ++i)
        {
            auto getSpecies = [this](float f) -> Species*
            {
                float currentSum = 0;
                for (Species* s : m_generation.m_species)
                {
                    currentSum += s->m_adjustedTotalScore;
                    if (currentSum > f)
                    {
                        return s;
                    }
                }

                assert(0);
                return nullptr;
            };

            Species* sp = getSpecies(GetRandomValue(adjustedScoreSum));
            genomesToInherit.push_back(GetInheritanceFromSpecies(*sp, newGenomes));
        }

        assert((int)genomesToInherit.size() <= GetNumGenomes());
    }

    // For the rest of population, perform inter-species cross over
    for (int i = (int)genomesToInherit.size(); i < GetNumGenomes(); ++i)
    {
        // Select random two genomes
        const Score* g1 = &SelectGenome(GetRandomValue(adjustedScoreSum), m_scores);
        const Score* g2 = g1;
        while (g1 == g2)
        {
            // Ensure we select different genomes
            g2 = &SelectGenome(GetRandomValue(adjustedScoreSum), m_scores);
        }

        // Ensure the genome at i1 has a higher fitness
        if (g1->m_fitness < g2->m_fitness)
        {
            std::swap(g1, g2);
        }

        // Cross over
        newGenomes.push_back(CrossOver(
            GetGenome(g1->m_index),
            g1->m_fitness,
            GetGenome(g2->m_index),
            g2->m_fitness));

#if TEST_PROTECT_NEWLY_CREATED_GENOME_BY_CROSS_OVER
        Genome& newGenome = newGenomes.back();

        const float fitness = Evaluate(newGenome);

        // Protect this species if it is the best among the generation
        // Note that we don't know which species this genome is yet, so we cannot update species protect flag etc
        if (fitness >= m_generation.m_species[0]->GetBestFitness())
        {
            newGenome.m_protect = true;
        }
#else
        const float fitness = 0.f;
#endif
        // We will use m_adjustedFitness == -1 later to indicate that this genome has newly generated
        genomesToInherit.push_back(Score{ fitness, -1.f, (int)newGenomes.size() - 1 });
    }

    assert((int)genomesToInherit.size() == GetNumGenomes());

    // Sort scores in descending order
    std::sort(genomesToInherit.rbegin(), genomesToInherit.rend());

    // Copy genomes to the next gen
    for (int i = 0; i < GetNumGenomes(); ++i)
    {
        int index = genomesToInherit[i].m_index;
        (*m_nextGenGenomesBuffer)[i] = genomesToInherit[i].m_adjustedFitness > 0 ? AccessGenome(index) : newGenomes[index];
    }

#ifdef _DEBUG
    if(m_config.m_enableSanityCheck)
    {
        // Clear genome indices stored in species
        for (Species* s : m_generation.m_species)
        {
            for (Score& sc : s->m_scores)
            {
                sc.m_index = -1;
            }

            s->m_bestScore.m_index = -1;
        }

        // Check if the best fitness is not regressed
        if (Evaluate((*m_generation.m_genomes)[0]) > Evaluate((*m_nextGenGenomesBuffer)[0]) &&
            m_generation.m_species[0]->ShouldProtectBest())
        {
            std::cout << "!!! Fitness Regressed !!!" << std::endl;
        }
    }
#endif

    // Swap buffers of genomes
    GenomeList tmp = m_generation.m_genomes;
    m_generation.m_genomes = m_nextGenGenomesBuffer;
    m_nextGenGenomesBuffer = tmp;

    // Increment generation
    m_generation.m_generationId++;
}

// Get genome and its score to inherit from a species
auto NEAT::GetInheritanceFromSpecies(Species& sp, std::vector<Genome>& newGenomes) -> Score
{
    // If there's only one genome in this species, just copy it.
    if (sp.GetNumGenomes() == 1)
    {
        return sp.m_scores[0];
    }

    // Select random two genomes
    const Score* g1 = &SelectGenome(GetRandomValue(sp.m_adjustedScoreEliminatedLows), sp.m_scores);
    const Score* g2 = g1;
    while (g1 == g2)
    {
        // Ensure we select different genomes
        g2 = &SelectGenome(GetRandomValue(sp.m_adjustedScoreEliminatedLows), sp.m_scores);
    }

    // Ensure g1 has a higher fitness
    if (g1->m_fitness < g2->m_fitness)
    {
        std::swap(g1, g2);
    }

    // Perform cross over
    newGenomes.push_back(CrossOver(
        GetGenome(g1->m_index),
        g1->m_fitness,
        GetGenome(g2->m_index),
        g2->m_fitness));

    Genome& newGenome = newGenomes.back();
    newGenome.m_species = sp.m_id;

#if TEST_PROTECT_NEWLY_CREATED_GENOME_BY_CROSS_OVER
    // Evaluate fitness of the new genome
    const float fitness = Evaluate(newGenome);

    // Update protect flag and best fitness of this species
    if (sp.ShouldProtectBest() && fitness >= sp.GetBestFitness())
    {
        newGenome.m_protect = true;
        if (fitness > sp.GetBestFitness())
        {
            AccessGenome(sp.GetBestGenome()).m_protect = false;
            sp.m_bestScore.m_fitness = fitness;

            // Note that we cannot update m_bestScore.m_index here
            // because index of this genome in the next generation hasn't been determined.
            // It will be confirmed after sorting all the genomes.
            // This shouldn't be a problem because all information we are using about
            // best score in this function is just fitness.
            // So updating just fitness should be sufficient.
        }
    }
#else
    const float fitness = 0.f;
#endif

    // We will use m_adjustedFitness == -1 later to indicate that this genome has newly generated
    return Score{ fitness, -1.f, (int)newGenomes.size() - 1 };
}

// Perform cross over operation over two genomes and generate a new genome
auto NEAT::CrossOver(const Genome& genome1, float fitness1, const Genome& genome2, float fitness2) -> Genome
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
    Genome child = CreateDefaultInitialGenome(true);

    RandomIntDistribution<int> randomBinary(0, 1);

    // Inherit connection genes from the two parents
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
                enabled, child);
        }
    }

    // Add remaining genes
    {
        auto AddRemainingGenes = [this, &randomBinary, &child](
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
                    // Disable gene at a certain probability when the parent's one is disabled
                    bool enabled = gene.m_enabled ? true : GetRandomProbability() >= m_config.m_geneDisablingRate;
                    TryAddConnection(gene, base, other, enabled, child);
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

    assert(CheckSanity(child));

    return child;
}

// Try to add the given connection to the child genome
void NEAT::TryAddConnection(const ConnectionGene& connection, const Genome* base, const Genome* other, bool enable, Genome& child) const
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

    // If either in or out node doesn't exist in the other parent,
    // we try to add the entire consecutive incompatible regions as a batch instead of just a single connection
    {
        bool hasIncompatibleNode = false;

        // NOTE: even if both inNode and outNode are incompatible, 
        // calling TryAddIncompatibleRegion only once should be sufficient because
        // we are adding all nodes and connections of consecutive incompatible region
        // hence else if below
        if (!other->HasNode(inNode))
        {
            TryAddIncompatibleRegion(inNode, base, other, child);
            hasIncompatibleNode = true;
        }
        else if (!other->HasNode(outNode))
        {
            TryAddIncompatibleRegion(outNode, base, other, child);
            hasIncompatibleNode = true;
        }
        if (hasIncompatibleNode)
        {
            return;
        }
    }

    // Make sure that child genome has both in and out nodes
    if (!child.HasNode(inNode))  child.AddNode(inNode);
    if (!child.HasNode(outNode)) child.AddNode(outNode);

    // Force to disable connection when it causes recurrent network
    if (!m_config.m_allowRecurrentNetwork && !CanAddConnectionWithoutRecurrent(child, inNode, outNode))
    {
        enable = false;
    }

    // Add new connection
    ConnectionGene newCon = connection;
    newCon.m_enabled = enable;
    child.AddConnection(newCon);
}

// Try to add a consecutive incompatible region to the child genome as a batch
void NEAT::TryAddIncompatibleRegion(const NodeGeneId incompatibleNode, const Genome* base, const Genome* other, Genome& child) const
{
    // Collect connections connected to incompatible nodes which exist only in base but not in other
    std::set<InnovationId> incompatibleConnections;
    GetIncompatibleRegionRecursive(incompatibleNode, base, other, incompatibleConnections);

    // Add connections
    for (InnovationId iid : incompatibleConnections)
    {
        assert(!child.HasConnection(iid));

        const ConnectionGene* c = GetConnectionGene(*base, iid);
        assert(c);

        const NodeGeneId inNode = c->m_inNode;
        const NodeGeneId outNode = c->m_outNode;

        // Make sure that child genome has both in and out nodes
        if (!child.HasNode(inNode))  child.AddNode(inNode);
        if (!child.HasNode(outNode)) child.AddNode(outNode);

        // Disable gene at a certain probability when the parent's one is disabled
        bool enabled = c->m_enabled ? true : GetRandomProbability() >= m_config.m_geneDisablingRate;;

        // Force to disable connection when it causes recurrent network
        if (!m_config.m_allowRecurrentNetwork && !CanAddConnectionWithoutRecurrent(child, inNode, outNode))
        {
            enabled = false;
        }

        // Add new connection
        ConnectionGene newCon = *c;
        newCon.m_enabled = enabled;
        child.AddConnection(newCon);
    }
}

// Collect connections connected to incompatible nodes which exist only in base but not in other
void NEAT::GetIncompatibleRegionRecursive(NodeGeneId current, const Genome* base, const Genome* other, std::set<InnovationId>& incompatibleConnections) const
{
    assert(base->HasNode(current));

    if (other->HasNode(current))
    {
        // Boundary of incompatible region
        return;
    }

    // Expand incompatible region recursively and collect connections
    for (InnovationId iid : base->GetIncommingConnections(current))
    {
        if (incompatibleConnections.find(iid) == incompatibleConnections.end())
        {
            incompatibleConnections.insert(iid);
            GetIncompatibleRegionRecursive(base->m_connectionGenes.at(iid).m_inNode, base, other, incompatibleConnections);
        }
    }
    for (InnovationId iid : base->GetOutgoingConnections(current))
    {
        if (incompatibleConnections.find(iid) == incompatibleConnections.end())
        {
            incompatibleConnections.insert(iid);
            GetIncompatibleRegionRecursive(base->m_connectionGenes.at(iid).m_outNode, base, other, incompatibleConnections);
        }
    }
}

// Evaluate a genome and return its fitness
float NEAT::Evaluate(const Genome& genome) const
{
    ++m_evaluationCount;

    return EvaluateImpl(genome);
}

// This function should be called prior to every evaluation
void NEAT::PrepareGenomeForEvaluation(const Genome& genome, std::unordered_map<NodeGeneId, float> initialValues) const
{
    // Reset nodes' value and state
    for (const auto& elem : genome.m_nodes)
    {
        NodeGeneId nodeId = elem.first;
        const Genome::Node& node = elem.second;
        NodeGeneType type = m_generation.m_nodeGenes.at(nodeId).m_type;
        if (type != NodeGeneType::Input && type != NodeGeneType::Bias)
        {
            node.m_state = NodeState::None;
            node.m_value = 0.f;
        }
        else
        {
            node.m_state = NodeState::Evaluated;
            if (initialValues.find(nodeId) != initialValues.end())
            {
                node.m_value = initialValues[nodeId];
            }
            else
            {
                node.m_value = 0.f;
            }
        }
    }
}

// Evaluate value of a node
float NEAT::EvaluateNode(const Genome& genome, NodeGeneId nodeId) const
{
    EvaluateNodeRecursive(genome, nodeId);
    const Genome::Node& node = genome.m_nodes.at(nodeId);
    assert(node.m_state == NodeState::Evaluated);
    return node.m_value;
}

// Evaluate value of nodes recursively
void NEAT::EvaluateNodeRecursive(const Genome& genome, NodeGeneId nodeId) const
{
    assert(genome.HasNode(nodeId));

    const Genome::Node& node = genome.m_nodes.at(nodeId);

    if (node.m_state == NodeState::Evaluated)
    {
        // Already evaluated this node
        return;
    }

    node.m_state = NodeState::Evaluating;

    float val = 0.f;

    // Evaluate all incoming connections of this node in order to evaluate this node
    for (auto innovId : genome.GetIncommingConnections(nodeId))
    {
        const auto connectionGene = GetConnectionGene(genome, innovId);
        assert(connectionGene);

        // Ignore disabled connection
        if (!connectionGene->m_enabled) continue;

        auto incomingNodeId = connectionGene->m_inNode;

        const Genome::Node& incomingNode = genome.m_nodes.at(incomingNodeId);
        if (incomingNode.m_state != NodeState::Evaluated)
        {
            // We've never evaluated this node yet. Evaluate it.

            // Check if we are already evaluating this node
            // if so, skip calling recursive function to avoid infinite loop
            if (incomingNode.m_state == NodeState::Evaluating) continue;

            // Evaluate the incoming node
            EvaluateNodeRecursive(genome, incomingNodeId);
        }

        // Calculate sum from all incoming connection
        val += incomingNode.m_value * connectionGene->m_weight;
    }

    // Apply activation function and store the result to the result map
    node.m_value = m_activationFuncs[m_generation.m_nodeGenes[nodeId].m_activationFuncId](val);
    node.m_state = NodeState::Evaluated;
}

// Set up node used for the initial network
void NEAT::SetupInitialNodeGenes()
{
    CreateNewNode(NodeGeneType::Input);
    CreateNewNode(NodeGeneType::Input);
    CreateNewNode(NodeGeneType::Output);
}

// Create default genome for the initial generation
auto NEAT::CreateDefaultInitialGenome(bool noConnections) -> Genome
{
    // Create two input nodes, one output node and two connections with random weight
    Genome genomeOut;

    NodeGeneId input1 = 0;
    NodeGeneId input2 = 1;
    NodeGeneId output = 2;
    genomeOut.AddNode(input1);
    genomeOut.AddNode(input2);
    genomeOut.AddNode(output);

    if (!noConnections)
    {
        Connect(genomeOut, input1, output, GetRandomWeight());
        Connect(genomeOut, input2, output, GetRandomWeight());
    }

    return genomeOut;
}

// Export genome as a neural network of smaller data representation which doesn't require information of the entire generation
auto NEAT::ExportAsNeuralNetwork(const Genome* genome) -> NN
{
    NN nn;

    // [TODO]: Do not copy unused activation functions
    for (auto& activation : m_activationFuncs)
    {
        nn.AddActivationFunction(activation);
    }

    std::unordered_map<NodeGeneId, int> nodeMap;
    std::unordered_map<InnovationId, int> connectionMap;

    std::vector<NN::Node> nodes;
    nodes.reserve(genome->GetNumNodes());

    for (const auto& elem : genome->m_nodes)
    {
        const NodeGeneId nodeId = elem.first;
        nodeMap[nodeId] = (int)nodes.size();
        NN::Node nnn;
        const NodeGene& node = GetNodeGene(nodeId);
        nnn.m_activationFunc = node.m_activationFuncId;
        nnn.m_type = node.m_type;
        nodes.push_back(nnn);
    }

    std::vector<NN::Connection> connections;
    connections.reserve(genome->GetNumConnections());

    for (const auto& elem : genome->m_connectionGenes)
    {
        const InnovationId innovId = elem.first;
        const ConnectionGene& con = elem.second;
        connectionMap[innovId] = (int)connections.size();
        NN::Connection nnc;
        nnc.m_weight = con.m_weight;
        nnc.m_inNode = nodeMap.at(con.m_inNode);
        nnc.m_outNode = nodeMap.at(con.m_outNode);
    }

    int i = 0;
    for (const auto& elem : genome->m_nodes)
    {
        const Genome::Node& node = elem.second;
        nodes[i].m_incomingConnections.reserve(node.m_incomings.size());
        for (const auto innovId : node.m_incomings)
        {
            nodes[i].m_incomingConnections.push_back(connectionMap[innovId]);
        }
    }

    return nn;
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

    // Check consistency of node links
    {
        for (auto elem : genome.m_nodes)
        {
            const Genome::Node& links = elem.second;
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

    // Check there is no dangling hidden nodes
    if(!m_config.m_removeConnectionsByMutation)
    {
        for (auto elem : genome.m_nodes)
        {
            if (m_generation.m_nodeGenes[elem.first].m_type == NodeGeneType::Hidden)
            {
                const Genome::Node& links = elem.second;
                // Hidden nodes must be connected to something in both directions
                // [TODO]: Verify this condition is reasonable. It's not explicitly stated in the original paper.
                if (links.m_incomings.size() == 0 || links.m_outgoings.size() == 0)
                {
                    return false;
                }
            }
        }
    }

    // Check if the network is not recurrent
    if (!m_config.m_allowRecurrentNetwork)
    {
        for (const auto con : genome.m_connectionGenes)
        {
            if (con.second.m_enabled && !CanAddConnectionWithoutRecurrent(genome, con.second.m_inNode, con.second.m_outNode))
            {
                return false;
            }
        }
    }

#endif
    return true;
}


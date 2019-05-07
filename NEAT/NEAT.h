#pragma once

#include <vector>
#include <functional>
#include <unordered_map>

// Custom hasher for std::pair
struct PairHash
{
    template <class T1, class T2>
    auto operator () (const std::pair<T1, T2>& p) const -> std::size_t
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2; // TODO: Too simple. This might not work very well as a hash.
    }
};

// Base class for NEAT
class NEAT
{
public:

    // Forward declarations
    struct NodeGene;
    struct ConnectionGene;
    struct Genome;

    // Type shortcuts
    using GenerationId = uint32_t;
    using NodeGeneId = uint32_t;
    using InnovationId = uint32_t;
    using ActivationFuncId = uint16_t;
    
    using ActivationFunc = std::function<float(float)>;
    using FitnessFunc = std::function<float(const Genome&)>;

    using NodeGeneList = std::vector<NodeGene>;
    using ConnectionGeneList = std::vector<ConnectionGene>;
    using NodeConnectionList = std::unordered_map<NodeGeneId, std::vector<InnovationId>>;

    // Definitions of invalid id
    static const NodeGeneId invalidGeneId = (NodeGeneId)-1;
    static const InnovationId invalidInnovationId = (InnovationId)-1;
    static const GenerationId invalidGenerationId = (GenerationId)-1;

    enum class NodeGeneType
    {
        Input,
        Output,
        Hidden
    };

    struct NodeGene
    {
        NodeGeneId id;
        NodeGeneType type;
    };

    struct ConnectionGene
    {
        InnovationId innovId;
        NodeGeneId inNode;
        NodeGeneId outNode;
        ActivationFuncId activationFuncId;
        float weight;
        bool enabled;
    };

    struct Genome
    {
        // List of node genes
        NodeGeneList nodeGenes;

        // List connection genes sorted by their innovation ids
        ConnectionGeneList connectionGenes;

        NodeConnectionList incomingConnectionList;
        NodeConnectionList outgoingConnectionList;
    };

    struct Generation
    {
        GenerationId generationId = 0;
        std::vector<Genome> genomes;
    };

    enum class DivercityProtectionMethod
    {
        None,
        Speciation,
        MorphologicalInnovationProtection,
    };

    // Settings of various parameters used in NEAT
    struct Configration
    {
        FitnessFunc fitnessFunction;
        ActivationFunc defaultActivateFunction;
        std::vector<ActivationFunc> activateFunctions;

        uint32_t numOrganismsInGeneration = 100;

        float weightMutationRate = .8f;
        float weightPerturbationRate = .9f;
        float geneDisablingRate = .75f;
        float crossOverRate = .75f;
        float interSpeciesMatingRate = .001f;
        float nodeAdditionRate = .03f;
        float connectionAdditionRate = .05f;

        bool useGlobalActivationFunc = true;

        // Indicates if NEAT allows to generate networks with cyclic connections
        // If false, generated networks are guaranteed to be feed forward
        bool allowCyclicNetwork = true;

        DivercityProtectionMethod divProtectMethod = DivercityProtectionMethod::Speciation;
    };

public:

    Configration config;

    // Initialize NEAT
    // This has to be called before gaining any generations
    // Returns the initial generation
    auto Initialize(const Configration& config) -> const Generation &;

    // Reset all the state in NEAT
    // After calling this function, Initialize has to be called in order to run NEAT again
    void Reset();

    // Gain the new generation
    auto GetNewGeneration() -> const Generation&;

protected:

    Generation generation;

    InnovationId currentInnovationId = 0;
    NodeGeneId currentNodeGeneId = 0;

    FitnessFunc fitnessFunc;

    ActivationFuncId defaultActivationFuncId = 0;
    std::vector<ActivationFunc> activationFuncs;

    using GenomeNodePair = std::pair<Genome&, NodeGeneId>;
    using GenomeConnectionPair = std::pair<Genome&, InnovationId>;
    using NodePair = std::pair<NodeGeneId, NodeGeneId>;

    std::unordered_map<InnovationId, std::vector<GenomeNodePair>> newlyAddedNodes;
    std::unordered_map<NodePair, std::vector<GenomeConnectionPair>, PairHash> newlyAddedConnections;

    // Add a new node at a random connection
    void AddNewNode(Genome& genome);

    // Add a new connection between random two nodes
    void AddNewConnection(Genome& genome);

    // Add a new connection between random two nodes allowing cyclic network
    void AddNewConnectionAllowingCyclic(Genome& genome);

    // Add a new connection between random two nodes without allowing cyclic network
    // Direction of the new connection is guaranteed to be one direction (distance from in node to an input node is smaller than the one of out node)
    void AddNewForwardConnection(Genome& genome);

    // Get shortest distance from a node to an input node
    // This function has to be called only when allowCyclicNetwork is false
    int GetNodeDepth(Genome& genome, NodeGeneId id) const;

    // Perform cross over operation over two genomes and generate a new genome
    // This function assumes that genome1 has a higher fitness value than genome2
    // Set sameFitness true when genome1 and genome2 have the same fitness values
    auto CrossOver(const Genome& genome1, const Genome& genome2, bool sameFitness) const -> NEAT::Genome;

    // Implementation of generating a new generation
    void GenerateNewGeneration();

    virtual float Evaluate(const Genome& genom) const;

    virtual auto CreateDefaultInitialGenome() const -> Genome;

    virtual int GetNumConnectionsInDefaultGenome() const;

    virtual int GetNumNodesInDefaultGenome() const;

    auto SelectRandomeGenome() -> Genome&;

    auto SelectRandomNodeGene(const std::vector<NodeGeneId>& genes) const -> NodeGeneId;

    auto SelectRandomNodeGene(const Genome& genome) const -> NodeGeneId;

    auto SelectRandomConnectionGene(const std::vector<InnovationId>& genes) const -> InnovationId;

    auto SelectRandomConnectionGene(const Genome& genome) const -> InnovationId;

    //void RemoveStaleGenes();

    void InitActivationFunc(ConnectionGene& gene) const;

    auto GetConnectionGene(Genome& genome, InnovationId innovId) -> ConnectionGene*;

    auto GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*;

    auto GetNodeGene(const Genome& genome, NodeGeneId id) const -> const NodeGene*;

    template <typename Gene, typename FuncType>
    static int FindGeneBinarySearch(const std::vector<Gene>& genes, uint32_t id, FuncType getIdFunc);
};

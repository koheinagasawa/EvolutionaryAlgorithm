#pragma once

#include <vector>
#include <functional>
#include <unordered_map>
#include <map>
#include <memory>

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
    struct Species;

    // Type shortcuts
    using GenerationId = uint32_t;
    using NodeGeneId = uint32_t;
    using InnovationId = uint32_t;
    using ActivationFuncId = uint16_t;
    using SpeciesId = uint16_t;

    using ActivationFunc = std::function<float(float)>;
    using FitnessFunc = std::function<float(const Genome&)>;

    using ConnectionGeneList = std::map<InnovationId, ConnectionGene>;
    using GenomeList = std::shared_ptr<std::vector<Genome>>;

    // Definitions of invalid id
    static const NodeGeneId invalidNodeGeneId = (NodeGeneId)-1;
    static const InnovationId invalidInnovationId = (InnovationId)-1;
    static const GenerationId invalidGenerationId = (GenerationId)-1;
    static const SpeciesId invalidSpeciesId = (SpeciesId)-1;

    enum class NodeGeneType
    {
        Input,
        Output,
        Hidden,
        Bias,
    };

    struct NodeGene
    {
        NodeGeneType type;
        ActivationFuncId activationFuncId;
    };

    struct ConnectionGene
    {
        InnovationId innovId;
        NodeGeneId inNode;
        NodeGeneId outNode;
        float weight;
        bool enabled;
    };

    struct Genome
    {
        struct Links
        {
            std::vector<InnovationId> incomings;
            std::vector<InnovationId> outgoings;
        };

        std::unordered_map<NodeGeneId, Links> nodeLinks;

        // List connection genes sorted by their innovation ids
        ConnectionGeneList connectionGenes;

        SpeciesId species;

        bool protect = false;
    };

    struct Score
    {
        float fitness;
        uint32_t index;
    };

    struct Species
    {
        SpeciesId id;
        Genome representative;
        std::vector<int> genomes;
        Score bestScore;
        float previousBestFitness = 0.f;
        int stagnantGenerationCount = 0;
        bool operator< (const Species& rhs) { return bestScore.fitness < rhs.bestScore.fitness; }
    };

    struct Generation
    {
        GenerationId generationId = 0;
        GenomeList genomes;
        std::vector<NodeGene> nodeGenes;
        std::vector<Species> species;
    };

    enum class DiversityProtectionMethod
    {
        None,
        Speciation,
        MorphologicalInnovationProtection,
    };

    // Settings of various parameters used in NEAT
    struct Configration
    {
        FitnessFunc fitnessFunction;
        ActivationFuncId defaultActivateFunctionId = 0;
        std::vector<ActivationFunc> activateFunctions;

        uint32_t numOrganismsInGeneration = 100;

        float weightMutationRate = .8f;
        float weightPerturbationRate = .9f;
        float geneDisablingRate = .75f;
        float crossOverRate = .75f;
        float interSpeciesMatingRate = .001f;
        float nodeAdditionRate = .03f;
        float connectionAdditionRate = .05f;
        float speciationDistThreshold = 3.f;

        bool enableCrossOver = true;

        bool useGlobalActivationFunc = true;

        // Indicates if NEAT allows to generate networks with cyclic connections
        // If false, generated networks are guaranteed to be feed forward
        bool allowCyclicNetwork = true;

        DiversityProtectionMethod diversityProtection = DiversityProtectionMethod::Speciation;

        bool enableSanityCheck = true;
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

    // Generate a new generation and return it
    auto GetNewGeneration() -> const Generation &;

    // Return the current generation
    auto GetCurrentGeneration() -> const Generation &;

    // Print fitness of each genome in the current generation
    void PrintFitness() const;

private:

    InnovationId currentInnovationId = 0;
    SpeciesId currentSpeciesId = 0;

    GenomeList genomeListBuffer = nullptr;

    // Evaluate all the current gen genomes
    std::vector<Score> scores;

protected:

    Generation generation;

    FitnessFunc fitnessFunc;

    ActivationFuncId defaultActivationFuncId = 0;
    std::vector<ActivationFunc> activationFuncs;

    using NodePair = std::pair<NodeGeneId, NodeGeneId>;

    std::unordered_map<NodePair, ConnectionGene, PairHash> innovationHistory;

    struct NodeAddedInfo
    {
        Genome& genome;
        NodeGeneId newNode;
        InnovationId oldConnection;
        InnovationId newConnection1;
        InnovationId newConnection2;
    };

    struct NewlyAddedConnection
    {
        Genome& genome;
        InnovationId innovId;
        NodePair nodes;
    };

    using NewlyAddedNodes = std::unordered_map<InnovationId, std::vector<NodeAddedInfo>>;

    bool isInitialized = false;

    inline auto GetCurrentInnovationId() const -> InnovationId { return currentInnovationId; }

    auto GetNewInnovationId() -> InnovationId;

    auto CreateNewNode(NodeGeneType type) -> NodeGeneId;

    auto Connect(Genome& genome, NodeGeneId inNode, NodeGeneId outNode, float weight) -> InnovationId;

private:

    // Add a new node at a random connection in the genome
    auto AddNewNode(Genome& genome) ->NodeAddedInfo;

    // Add a new connection between random two nodes in the genome
    void AddNewConnection(Genome& genome);

    // Add a new connection between random two nodes in the genome allowing cyclic network
    void AddNewConnectionAllowingCyclic(Genome& genome);

    // Add a new connection between random two nodes in the genome without allowing cyclic network
    // Direction of the new connection is guaranteed to be forward (distance from the input layer to in-node is smaller than the one for out-node)
    void AddNewForwardConnection(Genome& genome);

    auto GetNodeCandidatesAsOutNodeOfNewConnection(const Genome& genome) const -> std::vector<NodeGeneId>;

    bool CheckCyclic(const Genome& genome, NodeGeneId srcNode, NodeGeneId targetNode) const;

    // Perform cross over operation over two genomes and generate a new genome
    // This function assumes that genome1 has a higher fitness value than genome2
    auto CrossOver(const Genome& genome1, float fitness1, const Genome& genome2, float fitness2) const -> Genome;

    // Implementation of generating a new generation
    void GenerateNewGeneration();

    void Mutate(Generation& generation);

    // Make sure that the same topological changes have the same id
    void EnsureUniqueGeneIndices(const NewlyAddedNodes& newNodes);

    void Speciation(Generation& g, std::vector<Score>& scores, std::vector<int>& genomesToCopy);

    // Calculate distance between two genomes based on their topologies and weights
    float CalculateDistance(const Genome& genome1, const Genome& genome2) const;

protected:

    // Evaluate a genome and return its fitness
    virtual float Evaluate(const Genome& genom) const;

    // Evaluate value of each node recursively
    void EvaluateRecursive(const Genome& genome, NodeGeneId nodeId, std::vector<NodeGeneId>& evaluatingNodes, std::unordered_map<NodeGeneId, float>& values) const;

    virtual void SetupInitialNodeGenes();

    // Create default genome for the initial generation
    virtual auto CreateDefaultInitialGenome() const -> Genome;

    auto SelectRandomeGenome() -> Genome&;

    auto SelectRandomNodeGene(const std::vector<NodeGeneId>& genes) const -> NodeGeneId;

    auto SelectRandomNodeGene(const Genome& genome) const -> NodeGeneId;

    auto SelectRandomConnectionGene(const std::vector<InnovationId>& genes) const -> InnovationId;

    auto SelectRandomConnectionGene(const Genome& genome) const -> InnovationId;

    //void RemoveStaleGenes();

    auto GetConnectionGene(Genome& genome, InnovationId innovId) const -> ConnectionGene*;

    auto GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*;

    bool CheckSanity(const Genome& genome) const;
};


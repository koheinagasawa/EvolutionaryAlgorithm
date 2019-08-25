#pragma once

#include <vector>
#include <random>
#include <functional>
#include <unordered_map>
#include <map>
#include <set>
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
    using ActivationFuncId = uint8_t;
    using SpeciesId = uint16_t;

    using ActivationFunc = std::function<float(float)>;
    using FitnessFunc = std::function<float(const Genome&)>;

    using ConnectionGeneList = std::map<InnovationId, ConnectionGene>;
    using GenomeList = std::shared_ptr<std::vector<Genome>>;

    // Definitions of invalid id
    static const NodeGeneId s_invalidNodeGeneId = (NodeGeneId)-1;
    static const InnovationId s_invalidInnovationId = (InnovationId)-1;
    static const GenerationId s_invalidGenerationId = (GenerationId)-1;
    static const SpeciesId s_invalidSpeciesId = (SpeciesId)-1;

    enum class NodeGeneType
    {
        Input,
        Output,
        Hidden,
        Bias
    };

    struct NodeGene
    {
        NodeGeneType m_type;
        ActivationFuncId m_activationFuncId;
    };

    struct ConnectionGene
    {
        InnovationId m_innovId;
        NodeGeneId m_inNode;
        NodeGeneId m_outNode;
        float m_weight;
        bool m_enabled;
    };

    struct Genome
    {
        struct Links
        {
            int m_numEnabledIncomings = 0;
            int m_numEnabledOutgoings = 0;
            std::vector<InnovationId> m_incomings;
            std::vector<InnovationId> m_outgoings;
        };

        inline bool HasNode(NodeGeneId nodeId) const;
        inline bool HasConnection(InnovationId innovId) const;
        inline void AddConnection(const ConnectionGene& c);
        inline void DisableConnection(InnovationId innovId);
        inline void AddNode(NodeGeneId);
        inline int GetNumConnections() const;
        inline int GetNumNodes() const;
        inline auto GetIncommingConnections(NodeGeneId nodeId) const -> const std::vector<InnovationId>&;
        inline auto GetOutgoingConnections(NodeGeneId nodeId) const -> const std::vector<InnovationId>&;

        std::unordered_map<NodeGeneId, Links> m_nodeLinks;

        // List of connection genes sorted by their innovation ids
        ConnectionGeneList m_connectionGenes;

        SpeciesId m_species = s_invalidSpeciesId;

        bool m_protect = false;
    };

    struct Score
    {
        float m_fitness;
        float m_adjustedFitness;
        int m_index;
        inline bool operator< (const Score& rhs) const;
    };

    struct Species
    {
        SpeciesId m_id;
        Genome m_representative;
        std::vector<Score> m_scores;
        Score m_bestScore;
        float m_adjustedTotalScore = 0.f;
        float m_adjustedScoreEliminatedLows = 0.f;
        float m_previousBestFitness = 0.f;
        int m_stagnantGenerationCount = 0;
        
        inline int GetNumGenomes() const;
        inline float GetBestFitness() const;
        inline auto GetBestGenome() const -> NodeGeneId;
        inline bool ShouldProtectBest() const;
    };

    struct Generation
    {
        GenerationId m_generationId = 0;
        GenomeList m_genomes = nullptr;
        std::vector<NodeGene> m_nodeGenes;
        std::vector<Species*> m_species;
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
        FitnessFunc m_fitnessFunction;
        ActivationFuncId m_defaultActivateFunctionId = 0;
        std::vector<ActivationFunc> m_activateFunctions;

        int m_numGenomesInGeneration = 100;

        float m_maximumWeight = 5.0f;
        float m_minimumWeight = -5.0f;

        float m_weightMutationRate = .8f;
        float m_weightPerturbationRate = .9f;
        float m_weightNewValueRate = .1f;
        float m_weightPerturbation = 0.05f;
        float m_geneDisablingRate = .75f;
        float m_crossOverRate = .75f;
        float m_interSpeciesMatingRate = .001f;
        float m_nodeAdditionRate = .03f;
        float m_connectionAdditionRate = .05f;
        float m_speciationDistThreshold = 3.0f;
        float m_weightScaleForDistance = 0.4f;
        float m_lowerGenomeEliminationRate = 0.2f;

        bool m_enableCrossOver = true;

        bool m_useGlobalActivationFunc = true;
        bool m_extinctStagnantSpecies = true;
        int m_numGenerationsToExtinctSpecies = 15;

        // Indicates if NEAT allows to generate networks with cyclic connections
        // If false, generated networks are guaranteed to be feed forward
        bool m_allowCyclicNetwork = true;

        DiversityProtectionMethod m_diversityProtection = DiversityProtectionMethod::Speciation;

        bool m_enableSanityCheck = true;
    };

protected:

    using NodePair = std::pair<NodeGeneId, NodeGeneId>;

    struct NodeAddedInfo
    {
        Genome& m_genome;
        NodeGeneId m_newNode;
        InnovationId m_oldConnection;
        InnovationId m_newConnection1;
        InnovationId m_newConnection2;
    };

    using NewlyAddedNodes = std::unordered_map<InnovationId, std::vector<NodeAddedInfo>>;

    template <typename T>
    using RandomIntDistribution = std::uniform_int_distribution<T>;
    template <typename T>
    using RandomRealDistribution = std::uniform_real_distribution<T>;

public:

    // Initialize NEAT
    // This has to be called before gaining any generations
    // Returns the initial generation
    auto Initialize(const Configration& config) -> const Generation &;

    // Reset all the state in NEAT
    // After calling this function, Initialize has to be called in order to run NEAT again
    void Reset();

    // Generate a new generation and return it
    auto GetNewGeneration(bool printFitness) -> const Generation &;

    // Return the current generation
    auto GetCurrentGeneration() -> const Generation &;

    // Print fitness of each genome in the current generation
    void PrintFitness() const;

    // Serialize generation as a json file
    void SerializeGeneration(const char* fileName) const;

protected:

    inline auto GetCurrentInnovationId() const -> InnovationId;

    // Increment innovation id and return the new id
    auto GetNewInnovationId()->InnovationId;

    // Create a new node of a type
    auto CreateNewNode(NodeGeneType type)->NodeGeneId;

    // Connect the two nodes and assign the weight
    auto Connect(Genome& genome, NodeGeneId inNode, NodeGeneId outNode, float weight)->InnovationId;

    // Evaluate a genome and return its fitness
    virtual float Evaluate(const Genome& genom) const;

    // Evaluate value of each node recursively
    void EvaluateRecursive(const Genome& genome, NodeGeneId nodeId, std::vector<NodeGeneId>& evaluatingNodes, std::unordered_map<NodeGeneId, float>& values) const;

    // Set up node used for the initial network
    virtual void SetupInitialNodeGenes();

    // Create default genome for the initial generation
    virtual auto CreateDefaultInitialGenome()->Genome;

    //void RemoveStaleGenes();

    inline auto SelectRandomeGenome()->Genome &;

    inline auto SelectRandomNodeGene(const std::vector<NodeGeneId>& genes) const->NodeGeneId;

    inline auto SelectRandomNodeGene(const Genome& genome) const->NodeGeneId;

    inline auto SelectRandomConnectionGene(const std::vector<InnovationId>& genes) const->InnovationId;

    inline auto SelectRandomConnectionGene(const Genome& genome) const->InnovationId;

    inline auto GetConnectionGene(Genome& genome, InnovationId innovId) const->ConnectionGene*;

    inline auto GetConnectionGene(const Genome& genome, InnovationId innovId) const -> const ConnectionGene*;

    inline auto GetNodeGene(NodeGeneId nodeGeneId) const -> const NodeGene&;

    inline float GetRandomWeight() const;

    inline float GetRandomProbability() const;

    static inline float GetRandomValue(float max);

    bool CheckSanity(const Genome& genome) const;

private:

    inline auto AccessGenome(int index) -> Genome&;
    inline auto GetGenome(int index) const -> const Genome&;
    inline int GetNumGenomes() const;
    inline int GetNumSpecies() const;

    // Add a new node at a random connection in the genome
    auto AddNewNode(Genome& genome)->NodeAddedInfo;

    // Add a new connection between random two nodes in the genome allowing cyclic network
    // If allowCyclic, direction of the new connection is guaranteed to be forward (distance from the input layer to in-node is smaller than the one for out-node)
    void AddNewConnection(Genome& genome, bool allowCyclic);

    // Return false if adding a connection between srcNode to targetNode makes the network cyclic
    bool CanAddConnectionWithoutCyclic(const Genome& genome, NodeGeneId srcNode, NodeGeneId targetNode) const;

    // Implementation of generating a new generation
    void GenerateNewGeneration(bool printFitness);

    // Apply mutation
    void Mutate();

    // Make sure that the same topological changes have the same id
    void EnsureUniqueGeneIndices(const NewlyAddedNodes& newNodes);

    // Perform speciation
    void Speciation();

    // Calculate distance between two genomes based on their topologies and weights
    float CalculateDistance(const Genome& genome1, const Genome& genome2) const;

    // Select genomes to the next generation
    void SelectGenomes();

    // Get genome and its score to inherit from a species
    auto GetInheritanceFromSpecies(Species& species, std::vector<Genome>& newGenomes)->Score;

    // Perform cross over operation over two genomes and generate a new genome
    auto CrossOver(const Genome& genome1, float fitness1, const Genome& genome2, float fitness2) const->Genome;

    // Try to add the given connection to the child genome
    void TryAddConnection(const ConnectionGene& connection, const Genome* base, const Genome* other, bool enable, Genome& child) const;

    // Try to add a consecutive incompatible region to the child genome as a batch
    void TryAddIncompatibleRegion(const NodeGeneId incompatibleNode, const Genome* base, const Genome* other, Genome& child) const;

    // Collect connections connected to nodes which exist only in base but not in other
    void GetIncompatibleRegionRecursive(NodeGeneId current, const Genome* base, const Genome* other, std::set<InnovationId>& incompatibleConnections) const;

    inline auto SelectGenome(float value, const std::vector<Score>& scores) -> const Score&;

public:

    Configration m_config;

protected:

    Generation m_generation;

    std::vector<Score> m_scores;

    FitnessFunc m_fitnessFunc;

    ActivationFuncId m_defaultActivationFuncId = 0;
    std::vector<ActivationFunc> m_activationFuncs;

    std::unordered_map<NodePair, InnovationId, PairHash> m_innovationHistory;

    bool m_isInitialized = false;

private:

    InnovationId m_currentInnovationId = 0;
    SpeciesId m_currentSpeciesId = 0;

    // Buffer for next gen genomes to swap with the current gen
    GenomeList m_nextGenGenomesBuffer = nullptr;

    static std::default_random_engine s_randomGenerator;

    friend class NEAT_UnitTest;
};

#include "NEAT.inl"

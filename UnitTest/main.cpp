#include "../NEAT/NEAT.h"

#include <cmath>
#include <cassert>
#include <random>

static std::default_random_engine randomGenerator;

class XORNEAT : public NEAT
{
protected:

    virtual
    float Evaluate(const Genome& genome) const
    {
        auto outputNode = GetNodeGene(genome, m_outputNode);
        assert(outputNode != nullptr);
        auto outputNodeId = outputNode->id;

        auto eval = [&genome, outputNodeId, this](bool input1, bool input2) -> float
        {
            // Initialize values
            std::unordered_map<NodeGeneId, float> values;
            values[m_biasNode] = 1.f;
            values[m_inputNode1] = input1 ? 1.f : -1.f;
            values[m_inputNode2] = input2 ? 1.f : -1.f;

            std::vector<NodeGeneId> evaluatingNodes;
            EvaluateRecursive(genome, outputNodeId, evaluatingNodes, values);
            return values[outputNodeId];
        };

        float score = 0.f;

        // Test 4 patterns of XOR
        score += eval(false, false) >= 0.5f ? 0.f : 1.f;
        score += eval(false, true)  >= 0.5f ? 1.f : 0.f;
        score += eval(true,  false) >= 0.5f ? 1.f : 0.f;
        score += eval(true,  true)  >= 0.5f ? 0.f : 1.f;

        return score * score;
    }

    virtual
    auto CreateDefaultInitialGenome() const->Genome
    {
        Genome genome;

        // There are three input nodes (two for XOR inputs and one for bias)
        // no hidden node and one output node
        genome.nodeGenes.reserve(4);

        auto& nodeGenes = genome.nodeGenes;

        // Input nodes
        nodeGenes.push_back(NodeGene{ 0, NodeGeneType::Input, defaultActivationFuncId });
        nodeGenes.push_back(NodeGene{ 1, NodeGeneType::Input, defaultActivationFuncId });

        // Bias nodes
        nodeGenes.push_back(NodeGene{ 2, NodeGeneType::Bias, defaultActivationFuncId });

        // Output node
        nodeGenes.push_back(NodeGene{ 3, NodeGeneType::Output, defaultActivationFuncId });

        genome.connectionGenes.reserve(3);
        auto& connections = genome.connectionGenes;

        auto distr = std::uniform_real_distribution<float>(-1.f, 1.f);
        auto addConnectionGene = [&genome, &connections, &distr, this](InnovationId innovId, NodeGeneId nodeId1, NodeGeneId nodeId2)
        {
            connections.push_back(ConnectionGene{ innovId, nodeId1, nodeId2, distr(randomGenerator), true });
            genome.outgoingConnectionList[nodeId1].push_back(innovId);
            genome.incomingConnectionList[nodeId2].push_back(innovId);
        };

        addConnectionGene(0, m_inputNode1, m_outputNode);
        addConnectionGene(1, m_inputNode2, m_outputNode);
        addConnectionGene(2, m_biasNode, m_outputNode);

        return genome;
    }

private:

    static const NodeGeneId m_inputNode1 = 0;
    static const NodeGeneId m_inputNode2 = 1;
    static const NodeGeneId m_biasNode   = 2;
    static const NodeGeneId m_outputNode = 3;
};

int main()
{
    NEAT::Configration config;
    config.useGlobalActivationFunc = true;
    config.allowCyclicNetwork = false;
    config.activateFunctions.push_back([](float f)
        {return 1 / (1 + exp(-4.9f * f)); });
    config.diversityProtection = NEAT::DiversityProtectionMethod::None;
    config.numOrganismsInGeneration = 150;
    XORNEAT neat;
    neat.Initialize(config);

    for (int i = 0; i < 10000; ++i)
    {
        neat.GetNewGeneration();
        if (i % 10 == 0)
        {
            neat.PrintFitness();
        }
    }

    //neat.AddNewNode(neat.generation.genomes[0]);
    //neat.AddNewCoonection(neat.generation.genomes[0]);
    //neat.AddNewNode(neat.generation.genomes[1]);
    //neat.AddNewCoonection(neat.generation.genomes[1]);
    //NEAT::Genome genom = neat.CrossOver(neat.generation.genomes[0], neat.generation.genomes[1], true);

    return 0;
}
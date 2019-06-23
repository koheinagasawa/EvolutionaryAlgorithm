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
        auto eval = [&genome, this](bool input1, bool input2) -> float
        {
            // Initialize values
            std::unordered_map<NodeGeneId, float> values;
            values[m_biasNode] = 1.f;
            values[m_inputNode1] = input1 ? 1.f : -1.f;
            values[m_inputNode2] = input2 ? 1.f : -1.f;

            std::vector<NodeGeneId> evaluatingNodes;
            EvaluateRecursive(genome, m_outputNode, evaluatingNodes, values);
            return values[m_outputNode];
        };

        float score = 0.f;

        // Test 4 patterns of XOR
        score += eval(false, false) >= 0.5f ? 0.f : 1.f;
        score += eval(false, true)  >= 0.5f ? 1.f : 0.f;
        score += eval(true,  false) >= 0.5f ? 1.f : 0.f;
        score += eval(true,  true)  >= 0.5f ? 0.f : 1.f;

        return score * score;
    }

    virtual void SetupInitialNodeGenes() override
    {
        // There are three input nodes (two for XOR inputs and one for bias)
        // no hidden node and one output node
        auto& nodeGenes = generation.nodeGenes;
        nodeGenes.resize(4);

        // Input nodes
        nodeGenes[0] = NodeGene{ NodeGeneType::Input, defaultActivationFuncId };
        nodeGenes[1] = NodeGene{ NodeGeneType::Input, defaultActivationFuncId };

        // Bias nodes
        nodeGenes[2] = NodeGene{ NodeGeneType::Bias, defaultActivationFuncId };

        // Output node
        nodeGenes[3] = NodeGene{ NodeGeneType::Output, defaultActivationFuncId };
    }

    virtual
    auto CreateDefaultInitialGenome() const -> Genome
    {
        Genome genome;

        NodeGeneId input1 = 0;
        NodeGeneId input2 = 1;
        NodeGeneId bias = 2;
        NodeGeneId output = 3;

        auto distr = std::uniform_real_distribution<float>(-1.f, 1.f);
        ConnectionGene gene1{ 0, input1, output, distr(randomGenerator), true };
        ConnectionGene gene2{ 1, input2, output, distr(randomGenerator), true };
        ConnectionGene gene3{ 2, bias,   output, distr(randomGenerator), true };
        genome.nodeLinks[input1].outgoings.push_back(0);
        genome.nodeLinks[input2].outgoings.push_back(1);
        genome.nodeLinks[bias].outgoings.push_back(2);
        genome.nodeLinks[output].incomings.push_back(0);
        genome.nodeLinks[output].incomings.push_back(1);
        genome.nodeLinks[output].incomings.push_back(2);
        genome.connectionGenes[0] = gene1;
        genome.connectionGenes[1] = gene2;
        genome.connectionGenes[2] = gene3;

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
    config.diversityProtection = NEAT::DiversityProtectionMethod::Speciation;
    config.numOrganismsInGeneration = 150;
    XORNEAT neat;
    neat.Initialize(config);

    for (int i = 0; i < 10000; ++i)
    {
        //neat.GetNewGeneration(true);
        neat.GetNewGeneration(i % 10 == 0);
    }

    //neat.AddNewNode(neat.generation.genomes[0]);
    //neat.AddNewCoonection(neat.generation.genomes[0]);
    //neat.AddNewNode(neat.generation.genomes[1]);
    //neat.AddNewCoonection(neat.generation.genomes[1]);
    //NEAT::Genome genom = neat.CrossOver(neat.generation.genomes[0], neat.generation.genomes[1], true);

    return 0;
}
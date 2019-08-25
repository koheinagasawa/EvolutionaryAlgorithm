#include "../NEAT/NEAT.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sstream>

class XORNEAT : public NEAT
{
public:

    bool Test(const Genome& genome) const
    {
        bool result = true;
        // Test 4 patterns of XOR
        result &= EvalImpl(genome, false, false) <= 0.5f;
        result &= EvalImpl(genome, false, true ) > 0.5f;
        result &= EvalImpl(genome, true,  false) > 0.5f;
        result &= EvalImpl(genome, true,  true ) <= 0.5f;

        return result;
    }

protected:

    float EvalImpl(const Genome& genome, bool input1, bool input2) const
    {
        // Initialize values
        std::unordered_map<NodeGeneId, float> values;
        values[m_biasNode] = 1.f;
        values[m_inputNode1] = input1 ? 1.f : 0.f;
        values[m_inputNode2] = input2 ? 1.f : 0.f;

        std::vector<NodeGeneId> evaluatingNodes;
        EvaluateRecursive(genome, m_outputNode, evaluatingNodes, values);
        return values[m_outputNode];
    }

    virtual
    float Evaluate(const Genome& genome) const override
    {
        float score = 0.f;

        // Test 4 patterns of XOR
        score += EvalImpl(genome, false, false);
        score += 1.0f - EvalImpl(genome, false, true);
        score += 1.0f - EvalImpl(genome, true, false);
        score += EvalImpl(genome, true, true);
        score = 4.0f - score;

        return score * score;
    }

    // Set up node used for the initial network
    virtual void SetupInitialNodeGenes() override
    {
        // There are three input nodes (two for XOR inputs and one for bias)
        // no hidden node and one output node
        CreateNewNode(NodeGeneType::Input);
        CreateNewNode(NodeGeneType::Input);
        CreateNewNode(NodeGeneType::Bias);
        CreateNewNode(NodeGeneType::Output);
    }

    // Create default genome for the initial generation
    virtual auto CreateDefaultInitialGenome() -> Genome override
    {
        // Create three input nodes (one of them is a bias), one output node 
        // and three connections with random weights

        Genome genomeOut;

        NodeGeneId input1 = 0;
        NodeGeneId input2 = 1;
        NodeGeneId bias = 2;
        NodeGeneId output = 3;

        genomeOut.AddNode(input1);
        genomeOut.AddNode(input2);
        genomeOut.AddNode(bias);
        genomeOut.AddNode(output);

        Connect(genomeOut, input1, output, GetRandomWeight());
        Connect(genomeOut, input2, output, GetRandomWeight());
        Connect(genomeOut, bias, output, GetRandomWeight());

        return genomeOut;
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
    config.m_useGlobalActivationFunc = true;
    config.m_allowCyclicNetwork = false;
    config.m_activateFunctions.push_back([](float f)
    {
        return 1.f / (1.f + exp(-4.9f * f)); 
    });
    config.m_diversityProtection = NEAT::DiversityProtectionMethod::Speciation;
    config.m_numGenomesInGeneration = 150;
    XORNEAT neat;
    neat.Initialize(config);

    std::string baseOutputDir = "Test/";
    auto serialize = [baseOutputDir, &neat](int i)
    {
        std::stringstream ss;
        ss << baseOutputDir << "Gen" << i << ".json";
        neat.SerializeGeneration(ss.str().c_str());
    };

    for (int i = 0; i < 10000; ++i)
    {
        if (i > 0 && i % 10 == 0)
        {
            serialize(i);
        }

        const NEAT::Generation g = neat.GetNewGeneration(true);
        if (neat.Test((*g.m_genomes)[0]))
        {
            std::cout << "Solution Found at Generation " << i + 1 << "!" << std::endl;
            serialize(i+1);
            break;
        }
    }

    return 0;
}
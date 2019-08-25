#include "../NEAT/NEAT.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sstream>
#include <fstream>

#include <filesystem>
#if __cplusplus < 201703L
namespace fs = std::experimental::filesystem;
#else
namespace fs = std::filesystem;
#endif

class XorNEAT : public NEAT
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

        return EvaluateNode(genome, m_outputNode, values);
    }

    virtual
    float Evaluate(const Genome& genome) const override
    {
        ++m_evaluationCount;

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

        genomeOut.AddNode(m_inputNode1);
        genomeOut.AddNode(m_inputNode2);
        genomeOut.AddNode(m_biasNode);
        genomeOut.AddNode(m_outputNode);

        Connect(genomeOut, m_inputNode1, m_outputNode, GetRandomWeight());
        Connect(genomeOut, m_inputNode2, m_outputNode, GetRandomWeight());
        Connect(genomeOut, m_biasNode, m_outputNode, GetRandomWeight());

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
    // Configuring NEAT
    NEAT::Configration config;
    config.m_useGlobalActivationFunc = true;
    config.m_allowCyclicNetwork = false;
    config.m_activateFunctions.push_back([](float f)
    {
        return 1.f / (1.f + exp(-4.9f * f)); 
    });
    config.m_diversityProtection = NEAT::DiversityProtectionMethod::Speciation;
    config.m_numGenomesInGeneration = 150;
    config.m_enableSanityCheck = false;

    // Create NEAT
    XorNEAT neat;

    // Serialize option
    bool serializeAll = false;
    int serializationInterval = 5;
    auto serializeGeneration = [&neat](std::string baseOutputDir, int i)
    {
        std::stringstream ss;
        ss << baseOutputDir << "Gen" << i << ".json";
        neat.SerializeGeneration(ss.str().c_str());
    };

    // Variables for performance investigation
    const int maxGeneration = 100;
    const int numRun = 100;
    int numFailed = 0;
    int totalGenerations = 0;
    int worstGenerations = 0;
    int totalNumHiddenNodes = 0;
    int totalNumNondisabledConnections = 0;
    int totalEvaluationCount = 0;
    int worstEvaluationCount = 0;

    for (int run = 0; run < numRun; ++run)
    {
        std::cout << "Starting Run" << run << "..." << std::endl;

        neat.Initialize(config);

        std::string genomesOutputDir;
        if(serializeAll)
        {
            std::stringstream ss;
            ss << "Run" << run << "/";
            genomesOutputDir = ss.str();
            fs::create_directories(genomesOutputDir);
        }

        int generation = 0;
        for (int i = 0; generation < maxGeneration; ++generation)
        {
            if (serializeAll && i % serializationInterval == 0)
            {
                serializeGeneration(genomesOutputDir, generation);
            }

            const NEAT::Generation g = neat.GetNewGeneration(false);
            const NEAT::Genome& bestGenome = (*g.m_genomes)[0];
            if (neat.Test(bestGenome))
            {
                const int numGeneration = neat.GetCurrentGeneration().m_generationId;

                std::cout << "Solution Found at Generation " << numGeneration << "!" << std::endl;

                if (serializeAll)
                {
                    serializeGeneration(genomesOutputDir, numGeneration);
                }

                // Get data for performance investigation
                totalGenerations += numGeneration;
                if (worstGenerations < numGeneration)
                {
                    worstGenerations = numGeneration;
                }
                totalNumHiddenNodes += bestGenome.GetNumNodes() - 4; // 4 is two inputs, one output and out bias
                totalNumNondisabledConnections += bestGenome.GetNumEnabledConnections();
                if (worstEvaluationCount < neat.m_evaluationCount)
                {
                    worstEvaluationCount = neat.m_evaluationCount;
                }
                totalEvaluationCount += neat.m_evaluationCount;

                break;
            }
        }

        if (generation == maxGeneration)
        {
            std::cout << "Failed!" << std::endl;
            ++numFailed;
        }
    }

    const float invNumSuccess = 1.0f / float(numRun - numFailed);

    // Output result
    std::stringstream ss;
    ss << "=============================" << std::endl;
    ss << "Average successful generation : " << totalGenerations * invNumSuccess << std::endl;
    ss << "Worst successful generation : " << worstGenerations << std::endl;
    ss << "Number of failed run : " << numFailed << std::endl;
    ss << "Average number of hidden nodes of solution genome : " << totalNumHiddenNodes * invNumSuccess << std::endl;
    ss << "Average number of non-disabled connections of solution genome : " << totalNumNondisabledConnections * invNumSuccess << std::endl;
    ss << "Average evaluation count : " << totalEvaluationCount * invNumSuccess << std::endl;
    ss << "Worst evaluation count : " << worstEvaluationCount << std::endl;
    ss << "=============================" << std::endl;
    std::cout << ss.str();
    std::ofstream ofs("result.txt");
    ofs << ss.str();
    ofs.close();

    return 0;
}
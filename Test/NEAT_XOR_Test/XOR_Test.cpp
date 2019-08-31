#include <NEAT/NEAT.h>

#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>
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
        result &= EvaluateImpl(genome, false, false) <= 0.5f;
        result &= EvaluateImpl(genome, false, true ) > 0.5f;
        result &= EvaluateImpl(genome, true,  false) > 0.5f;
        result &= EvaluateImpl(genome, true,  true ) <= 0.5f;

        return result;
    }

protected:

    float EvaluateImpl(const Genome& genome, bool input1, bool input2) const
    {
        // Initialize values
        std::unordered_map<NodeGeneId, float> values;
        values[m_biasNode] = 1.f;
        values[m_inputNode1] = input1 ? 1.f : 0.f;
        values[m_inputNode2] = input2 ? 1.f : 0.f;
        PrepareGenomeForEvaluation(genome, values);

        return EvaluateNode(genome, m_outputNode);
    }

    virtual
    float EvaluateImpl(const Genome& genome) const override
    {
        ++m_evaluationCount;

        float score = 0.f;

        // Test 4 patterns of XOR
        score += EvaluateImpl(genome, false, false);
        score += 1.0f - EvaluateImpl(genome, false, true);
        score += 1.0f - EvaluateImpl(genome, true, false);
        score += EvaluateImpl(genome, true, true);
        score = 4.0f - score;

        return score * score;
    }

    // Set up node used for the initial network
    virtual void SetupInitialNodeGenes() override
    {
        // There are three input nodes (two for XOR inputs and one for bias),
        // no hidden node and one output node
        m_inputNode1 = CreateNewNode(NodeGeneType::Input);
        m_inputNode2 = CreateNewNode(NodeGeneType::Input);
        m_biasNode   = CreateNewNode(NodeGeneType::Bias);
        m_outputNode = CreateNewNode(NodeGeneType::Output);
    }

    // Create default genome for the initial generation
    virtual auto CreateDefaultInitialGenome(bool noConnections) -> Genome override
    {
        // Create three input nodes (one of them is a bias), one output node 
        // and three connections with random weights

        Genome genomeOut;

        genomeOut.AddNode(m_inputNode1);
        genomeOut.AddNode(m_inputNode2);
        genomeOut.AddNode(m_biasNode);
        genomeOut.AddNode(m_outputNode);

        if (!noConnections)
        {
            Connect(genomeOut, m_inputNode1, m_outputNode, GetRandomWeight());
            Connect(genomeOut, m_inputNode2, m_outputNode, GetRandomWeight());
            Connect(genomeOut, m_biasNode, m_outputNode, GetRandomWeight());
        }

        return genomeOut;
    }

private:

    NodeGeneId m_inputNode1;
    NodeGeneId m_inputNode2;
    NodeGeneId m_biasNode;
    NodeGeneId m_outputNode;
};

int main()
{
    // Configuring NEAT
    NEAT::Configration config;
    config.m_useGlobalActivationFunc = true;
    config.m_activateFunctions.push_back([](float f)
    {
        return 1.f / (1.f + exp(-4.9f * f)); 
    });
    config.m_diversityProtection = NEAT::DiversityProtectionMethod::Speciation;
    config.m_numGenomesInGeneration = 150;
    config.m_allowRecurrentNetwork = false;
    config.m_enableSanityCheck = false;

    // Create NEAT
    XorNEAT neat;

    // Serialize option
    bool serializeGenomes = false;
    int serializationGenerationInterval = 10;
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
        if(serializeGenomes)
        {
            std::stringstream ss;
            ss << "Results/Run" << run << "/";
            genomesOutputDir = ss.str();
            fs::create_directories(genomesOutputDir);
        }

        int i = 0;
        for (; i < maxGeneration; ++i)
        {
            const NEAT::Generation& g = neat.GetNewGeneration(false);
            const int numGeneration = g.m_generationId;

            const NEAT::Genome& bestGenome = (*g.m_genomes)[0];
            if (neat.Test(bestGenome))
            {
                std::cout << "Solution Found at Generation " << numGeneration << "!" << std::endl;

                if (serializeGenomes)
                {
                    serializeGeneration(genomesOutputDir, numGeneration);
                }

                // Get data for performance investigation
                totalGenerations += numGeneration;
                if (worstGenerations < numGeneration)
                {
                    worstGenerations = numGeneration;
                }
                totalNumHiddenNodes += bestGenome.GetNumNodes() - 4; // 4 is two inputs, one output and one bias
                totalNumNondisabledConnections += bestGenome.GetNumEnabledConnections();
                if (worstEvaluationCount < neat.m_evaluationCount)
                {
                    worstEvaluationCount = neat.m_evaluationCount;
                }
                totalEvaluationCount += neat.m_evaluationCount;

                break;
            }

            if (serializeGenomes && numGeneration % serializationGenerationInterval == 0)
            {
                serializeGeneration(genomesOutputDir, numGeneration);
            }
        }

        if (i == maxGeneration)
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
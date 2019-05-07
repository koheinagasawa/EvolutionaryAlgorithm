#include "../NEAT/NEAT.h"
#include <cmath>

class XORNEAT : public NEAT
{
protected:

    virtual
    float Evaluate(const Genome& genom) const
    {

    }

    virtual
    auto CreateDefaultInitialGenome() const->Genome
    {

    }
};

int main()
{
    NEAT::Configration config;
    config.useGlobalActivationFunc = true;
    config.activateFunctions.push_back([](float f)
        {return 1 / (1 + exp(-4.9f * f)); });
    //config.numOrganismsInGeneration = 2;
    NEAT neat;
    neat.Initialize(config);

    for (int i = 0; i < 10000; ++i)
    {
        neat.GetNewGeneration();
    }

    //neat.AddNewNode(neat.generation.genomes[0]);
    //neat.AddNewCoonection(neat.generation.genomes[0]);
    //neat.AddNewNode(neat.generation.genomes[1]);
    //neat.AddNewCoonection(neat.generation.genomes[1]);
    //NEAT::Genome genom = neat.CrossOver(neat.generation.genomes[0], neat.generation.genomes[1], true);

    return 0;
}
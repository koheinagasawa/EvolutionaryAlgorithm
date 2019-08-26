"""
Create and draw a star with varying node properties.
"""

from pygraphviz import *
import json
import os

def OutputGenomes(inputJsonFile, numGenomes, outImages, outDots, imageOutDir = "", dotOutDir = ""):

    inputFileBase = os.path.dirname(inputJsonFile) + "/"

    if not os.path.exists(inputFileBase):
        os.makedirs(inputFileBase)

    with open(inputJsonFile, 'r') as f:
        obj = json.load(f)

    generationId = obj["GenerationId"]
    generationOutDir = ""
    if (outImages == True and len(imageOutDir) == 0) or (outDots == True and len(dotOutDir) == 0):
        generationOutDir = inputFileBase + "Gen" + str(generationId) + "/"
        if not os.path.exists(generationOutDir):
            os.makedirs(generationOutDir)

    if outImages:
        if len(imageOutDir) == 0:
            imageOutDir = generationOutDir + "images/"
        if not os.path.exists(imageOutDir):
            os.makedirs(imageOutDir)

    if outDots:
        if len(dotOutDir) == 0:
            dotOutDir = generationOutDir + "dots/"
        if not os.path.exists(dotOutDir):
            os.makedirs(dotOutDir)

    i = 0
    for genome in obj["Genomes"]:
        i += 1
        if numGenomes > 0 and i > numGenomes:
            break
            
        g = AGraph(directed=True)

        # set some default node attributes
        g.node_attr['style']='filled'
        g.node_attr['shape']='circle'
        g.node_attr['fixedsize']='true'
        g.node_attr['fontcolor']='#FFFFFF'

        for node in genome["Nodes"]:
            nodeType = node["Type"]
            nodeColor = "white"
            if nodeType == "INPUT":
                nodeColor = "blue"
            elif nodeType == "OUTPUT":
                nodeColor = "red"
            elif nodeType == "BIAS":
                nodeColor = "green"
            else:
                nodeColor = "black"
            g.add_node(node["Id"], color=nodeColor)

        for connection in genome["Connections"]:
            id = connection["InnovationId"]
            inNode = connection["InNode"]
            outNode = connection["OutNode"]
            weight = connection["Weight"]
            enabled = connection["Enabled"]
            g.add_edge(inNode, outNode, label = str(id) + "\n" + str(weight))

            if not enabled:
                e = g.get_edge(inNode, outNode)
                e.attr['style'] = 'dotted'

        fileNameBase = "Gen" + str(generationId) + "_Genome" + str(genome["Index"]) + "_Sp" + str(genome["SpeciesId"])
        
        if outDots:
            g.write(dotOutDir + fileNameBase + ".dot") # write to simple.dot

        if outImages:
            g.draw(imageOutDir + fileNameBase + '.png',prog="dot") # draw to png using circo

        g.close()

    return [imageOutDir, dotOutDir]


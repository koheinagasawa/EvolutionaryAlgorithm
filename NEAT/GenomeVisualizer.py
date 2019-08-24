"""
Create and draw a star with varying node properties.
"""

from pygraphviz import *
import json
import os

def OutputGenomes(inputJsonFile):

    inputFileBase = os.path.basename(os.path.dirname(inputJsonFile)) + "/"

    if not os.path.exists(inputFileBase):
        os.makedirs(inputFileBase)

    with open(inputJsonFile, 'r') as f:
        obj = json.load(f)

    generationOutDir = inputFileBase + "Gen" + str(obj["GenerationId"]) + "/"
    if not os.path.exists(generationOutDir):
        os.makedirs(generationOutDir)

    imageOutDir = generationOutDir + "images/"
    if not os.path.exists(imageOutDir):
        os.makedirs(imageOutDir)

    dotOutDir = generationOutDir + "dots/"
    if not os.path.exists(dotOutDir):
        os.makedirs(dotOutDir)

    for genome in obj["Genomes"]:
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

        fileNameBase = "Genome" + str(genome["Index"]) + "[Sp" + str(genome["SpeciesId"]) + "]"
        g.write(dotOutDir + fileNameBase + ".dot") # write to simple.dot
        g.draw(imageOutDir + fileNameBase + '.png',prog="dot") # draw to png using circo

# for elem in obj:
#     print(elem)
#     print(obj[elem])

# A=AGraph()

# # set some default node attributes
# A.node_attr['style']='filled'
# A.node_attr['shape']='circle'
# A.node_attr['fixedsize']='true'
# A.node_attr['fontcolor']='#000000'

# # make a star in shades of red
# for i in range(16):
#     A.add_edge(0,i)
#     e=A.get_edge(0,i)
#     e.attr['label']="edge-"
#     n=A.get_node(i)
#     n.attr['fillcolor']="#%2x0000"%(i*16)
#     n.attr['label']="test"
#     # n.attr['height']="%s"%(i/16.0+0.5)
#     # n.attr['width']="%s"%(i/16.0+0.5)

# print(A.string()) # print to screen
# A.write("star.dot") # write to simple.dot
# print("Wrote star.dot")
# A.draw('star.png',prog="circo") # draw to png using circo
# print("Wrote star.png")
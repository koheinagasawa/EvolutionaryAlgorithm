#include "UnstructuredNeuralNetwork.h"

#include <stack>
#include <cassert>

void UnstructuredNeuralNetwork::Init(const std::vector<Node>& nodes, const std::vector<Connection>& connections)
{
    m_nodes = nodes;
    m_connections = connections;

    m_inputNodes.clear();
    m_outputNodes.clear();

    for (int i = 0; i < (int)m_nodes.size(); ++i)
    {
        if (m_nodes[i].m_type == NodeType::Input)
        {
            m_inputNodes.push_back(i);
        }
        else if (m_nodes[i].m_type == NodeType::Output)
        {
            m_outputNodes.push_back(i);
        }
    }

    Reset();

    assert(SanityCheck());
}

void UnstructuredNeuralNetwork::AddNodes(int numNodes, ActivationFuncId activation, NodeType type)
{
    if (numNodes == 0) return;

    m_nodes.reserve(m_nodes.size() + numNodes);
    for (int i = 0; i < numNodes; ++i)
    {
        m_nodes.push_back(Node{ 0.0f, NodeState::None, std::vector<ConnectionId>(), activation, type });
    }

    if (type == NodeType::Input || type == NodeType::Bias)
    {
        m_inputNodes.reserve(m_inputNodes.size() + numNodes);
        for (int i = 0; i < numNodes; ++i)
        {
            m_inputNodes.push_back(i);
        }
    }
    else if (type == NodeType::Output)
    {
        m_outputNodes.reserve(m_outputNodes.size() + numNodes);
        for (int i = 0; i < numNodes; ++i)
        {
            m_outputNodes.push_back(i);
        }
    }

    assert(SanityCheck());
}

void UnstructuredNeuralNetwork::Connect(NodeId inNode, NodeId outNode)
{
    assert(isValidNode(inNode) && isValidNode(outNode));

    m_connections.push_back(Connection{ 0.0f, inNode, outNode });

    m_nodes[outNode].m_incomingConnections.push_back((int)m_connections.size() - 1);

    assert(SanityCheck());
}

void UnstructuredNeuralNetwork::Clear()
{
    m_activationFuncs.clear();
    m_nodes.clear();
    m_inputNodes.clear();
    m_outputNodes.clear();
    m_connections.clear();
}

auto UnstructuredNeuralNetwork::AddActivationFunction(ActivationFunc func) -> ActivationFuncId
{
    m_activationFuncs.push_back(func);
    return (uint8_t)(m_activationFuncs.size() - 1);
}

void UnstructuredNeuralNetwork::Reset()
{
    for (int i = 0; i < (int)m_nodes.size(); ++i)
    {
        m_nodes[i].m_value = 0.0f;
        m_nodes[i].m_state = NodeState::None;
    }
}

void UnstructuredNeuralNetwork::SetInputNodeValues(const std::vector<float>& values)
{
    assert(values.size() == m_inputNodes.size());

    for (int i = 0; i < (int)values.size(); ++i)
    {
        NodeId nodeId = m_inputNodes[i];
        Node& node = m_nodes[nodeId];
        assert(node.m_type == NodeType::Input || node.m_type == NodeType::Bias);
        node.m_value = values[i];
        node.m_state = NodeState::Evaluated;
    }
}

void UnstructuredNeuralNetwork::SetNodeValues(const std::vector<float>& values)
{
    assert(values.size() == m_nodes.size());

    for (int i = 0; i < (int)values.size(); ++i)
    {
        NodeId nodeId = m_inputNodes[i];
        Node& node = m_nodes[nodeId];
        node.m_value = values[i];
        if (node.m_type == NodeType::Input || node.m_type == NodeType::Bias)
        {
            node.m_state = NodeState::Evaluated;
        }
    }
}

auto UnstructuredNeuralNetwork::GetOutputNodeValues() const -> std::vector<float>
{
    std::vector<float> out;
    if (m_outputNodes.size() == 0)
    {
        return out;
    }

    out.reserve(m_outputNodes.size());

    for (NodeId node : m_outputNodes)
    {
        out.push_back(m_nodes[node].m_value);
    }

    return out;
}

auto UnstructuredNeuralNetwork::GetNodeValues() const -> std::vector<float>
{
    std::vector<float> out;
    if (m_nodes.size() == 0)
    {
        return out;
    }

    out.reserve(m_nodes.size());

    for (const Node& node : m_nodes)
    {
        out.push_back(node.m_value);
    }

    return out;
}

void UnstructuredNeuralNetwork::Evaluate()
{
    for (NodeId node : m_outputNodes)
    {
        EvaluateNodeRecursive(node);
    }
}

void UnstructuredNeuralNetwork::EvaluateNodeRecursive(NodeId nodeId)
{
    assert(isValidNode(nodeId));
    Node& node = m_nodes[nodeId];

    if (node.m_state == NodeState::Evaluated)
    {
        // Already evaluated this node
        return;
    }

    float val = 0.f;

    // Evaluate all incoming connections of this node in order to evaluate this node
    for (auto connectionId : node.m_incomingConnections)
    {
        assert(isValidConnection(connectionId));

        auto incomingNodeId = m_connections[connectionId].m_inNode;
        assert(isValidNode(incomingNodeId));
        Node& inNode = m_nodes[incomingNodeId];

        if (inNode.m_state != NodeState::Evaluated)
        {
            // We've never evaluated this node yet. Evaluate it.

            // Check if we are already evaluating this node
            // if so, skip calling recursive function to avoid infinite loop
            if (inNode.m_state == NodeState::Evaluating) continue;

            inNode.m_state = NodeState::Evaluating;

            // Evaluate the incoming node
            EvaluateNodeRecursive(incomingNodeId);
        }

        // Calculate sum from all incoming connection
        val += inNode.m_value * m_connections[connectionId].m_weight;
    }

    // Apply activation function and store the result to the result map
    node.m_value = m_activationFuncs[node.m_activationFunc](val);
    node.m_state = NodeState::Evaluated;
}

bool UnstructuredNeuralNetwork::IsRecurrentNetwork() const
{
    for (const Connection& con : m_connections)
    {
        if (CanReachFromSourceToTarget(con.m_outNode, con.m_inNode))
        {
            return true;
        }
    }

    return false;
}

bool UnstructuredNeuralNetwork::CanReachFromSourceToTarget(NodeId srcNode, NodeId targetNode) const
{
    if (srcNode == targetNode)
    {
        return true;
    }

    std::vector<int> flag;
    flag.resize((int)m_nodes.size() / sizeof(int) + 1, 0);

    // Test if we can reach to the targetNode from srcNode by following connections reversely
    std::stack<NodeId> stack;
    stack.push(srcNode);
    while (!stack.empty())
    {
        const NodeId node = stack.top();
        stack.pop();

        // Test all incoming connections of this node
        for (const ConnectionId cid : m_nodes[node].m_incomingConnections)
        {
            const NodeId inNode = m_connections[cid].m_inNode;

            if (inNode == targetNode)
            {
                // Reached to the target node
                // The new connection will make the network recurrent
                return true;
            }

            const int index = inNode / sizeof(int);
            const int offset = inNode % sizeof(int);
            // Add this node to the stack if we haven't yet
            if (((flag[index] >> offset) & 1) == 0)
            {
                stack.push(inNode);
                flag[index] |= 1 << offset;
            }
        }
    }

    return false;
};

bool UnstructuredNeuralNetwork::SanityCheck() const
{
    return true;
}

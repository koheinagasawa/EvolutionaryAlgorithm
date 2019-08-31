#pragma once

#include <vector>
#include <functional>

class UnstructuredNeuralNetwork
{
public:

    struct Node;
    struct Connection;

    using NodeId = uint32_t;
    using ConnectionId = uint32_t;
    using ActivationFunc = std::function<float(float)>;
    using ActivationFuncId = uint8_t;

    static const NodeId invalidNodeId = (NodeId)-1;

    enum class NodeType
    {
        Input,
        Output,
        Hidden,
        Bias
    };

    enum class NodeState
    {
        None,
        Evaluating,
        Evaluated
    };

    struct Node
    {
        float m_value = 0.0f;
        NodeState m_state = NodeState::None;
        std::vector<ConnectionId> m_incomingConnections;
        ActivationFuncId m_activationFunc;
        NodeType m_type;
    };

    struct Connection
    {
        mutable float m_weight;
        NodeId m_inNode;
        NodeId m_outNode;
    };

    void Init(const std::vector<Node>& nodes, const std::vector<Connection>& connections);

    void AddNodes(int numNodes, ActivationFuncId activation, NodeType type);

    void Connect(NodeId inNode, NodeId outNode);

    void Clear();

    auto AddActivationFunction(ActivationFunc func) -> ActivationFuncId;

    inline auto GetInputNodes() const -> const std::vector<NodeId>& { return m_inputNodes; }
    inline auto GetOutputNodes() const -> const std::vector<NodeId>& { return m_outputNodes; }

    void Reset();

    void SetInputNodeValues(const std::vector<float>& values);
    void SetNodeValues(const std::vector<float>& values);

    auto GetOutputNodeValues() const -> std::vector<float>;
    auto GetNodeValues() const -> std::vector<float>;

    void Evaluate();

    bool IsRecurrentNetwork() const;

    bool SanityCheck() const;

protected:

    void EvaluateNodeRecursive(NodeId nodeId);

    bool CanReachFromSourceToTarget(NodeId srcNode, NodeId targetNode) const;

    inline bool isValidNode(NodeId node) const { return node >= 0 && node < (int)m_nodes.size(); }
    inline bool isValidConnection(ConnectionId connection) const { return connection >= 0 && connection < (int)m_connections.size(); }

protected:

    std::vector<ActivationFunc> m_activationFuncs;

    std::vector<Node> m_nodes;
    std::vector<NodeId> m_inputNodes;
    std::vector<NodeId> m_outputNodes;
    std::vector<Connection> m_connections;
};
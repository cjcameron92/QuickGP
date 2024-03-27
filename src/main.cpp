#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <random>
#include <sstream>


std::vector<std::function<double(double, double)>> ops = {
        [](double a, double b) { return a + b; },
        [](double a, double b) { return a - b; },
        [](double a, double b) { return a * b; },
        [](double a, double b) { return (b != 0) ? a / b : 1.0; }
};

std::vector<char> symbols = {'+', '-', '*', '/'};
std::map<std::string, double> vars = {{"x", 10.0}};

class Node {
public:
    virtual double evaluate(const std::map<std::string, double>& vars) = 0;
    virtual std::string toString() const = 0;
    virtual std::unique_ptr<Node> clone() const = 0;
    virtual ~Node() {}
};


class ConstantNode : public Node {
    double value;
public:
    ConstantNode(double value) : value(value) {}
    double evaluate(const std::map<std::string, double>&) override {
        return value;
    }
    std::unique_ptr<Node> clone() const override {
        return std::make_unique<ConstantNode>(*this);
    }
    std::string toString() const override {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }
};

class OperatorNode : public Node {
    std::function<double(double, double)> op;
    char opSymbol;
public:
    OperatorNode(std::function<double(double, double)> op, std::unique_ptr<Node> l, std::unique_ptr<Node> r, char symbol)
            : op(op), left(std::move(l)), right(std::move(r)), opSymbol(symbol) {}
    double evaluate(const std::map<std::string, double>& vars) override {
        return op(left->evaluate(vars), right->evaluate(vars));
    }
    std::unique_ptr<Node> clone() const override {
        return std::make_unique<OperatorNode>(op, left->clone(), right->clone(), opSymbol);
    }

    std::string toString() const override {
        return "(" + left->toString() + " " + opSymbol + " " + right->toString() + ")";
    }

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node* getLeft() const { return left.get(); }
    Node* getRight() const { return right.get(); }
    void setLeft(std::unique_ptr<Node> newLeft) { left = std::move(newLeft); }
    void setRight(std::unique_ptr<Node> newRight) { right = std::move(newRight); }
};

class TerminalNode : public Node {
    std::string name;
public:
    TerminalNode(const std::string& name) : name(name) {}
    double evaluate(const std::map<std::string, double>& vars) override {
        auto it = vars.find(name);
        return it != vars.end() ? it->second : 0.0;
    }
    std::unique_ptr<Node> clone() const override {
        return std::make_unique<TerminalNode>(*this);
    }
    std::string toString() const override {
        return name;
    }
};

class Tree {
public:
    std::unique_ptr<Node> root;

    Tree(std::unique_ptr<Node> rootNode) : root(std::move(rootNode)) {}

    double evaluate(const std::map<std::string, double>& vars) const {
        return root->evaluate(vars);
    }


    double func(double x) const {
        return std::pow(x, 4) + std::pow(x, 3) + std::pow(x, 2) + x;
    }

    // Modify the fitness method to use func instead of evaluate
    double fitness(std::map<std::string, double>& vars, double targetValue) const {
        std::vector<double> points = {-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
        double squareMeanError = 0;
        for (double point : points) {
            vars["x"] = point;
            double result = evaluate(vars);
            squareMeanError += std::pow(result - func(point), 2);
        }
        return squareMeanError / points.size();
    }

    std::string toString() const {
        return root->toString();
    }

    void collectNodes(std::unique_ptr<Node>& node, std::vector<std::unique_ptr<Node>*>& nodes) {
        if (!node) return;
        nodes.push_back(&node);
        if (auto opNode = dynamic_cast<OperatorNode*>(node.get())) {
            collectNodes(opNode->left, nodes);
            collectNodes(opNode->right, nodes);
        }
    }

    std::vector<std::unique_ptr<Node>*> getAllNodes() {
        std::vector<std::unique_ptr<Node>*> nodes;
        collectNodes(root, nodes);
        return nodes;
    }
};

std::vector<std::string> terminals = {"x"};

std::unique_ptr<Node> createRandomTree(int depth, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(-1, 1);
    std::uniform_int_distribution<> opDis(0, ops.size() - 1);
    std::uniform_int_distribution<> termDis(0, terminals.size() - 1);
    std::uniform_int_distribution<> nodeTypeDis(0, 2);

    if (depth == 1) {
        if (nodeTypeDis(gen) % 2 == 0 && !terminals.empty()) {
            return std::make_unique<TerminalNode>(terminals[termDis(gen)]);
        } else {
            double sampledValue = dis(gen);
            int roundedValue = (sampledValue > 0) - (sampledValue < 0);
            return std::make_unique<ConstantNode>(roundedValue);
        }
    } else {
        int index = opDis(gen);
        auto left = createRandomTree(depth - 1, gen);
        auto right = createRandomTree(depth - 1, gen);
        return std::make_unique<OperatorNode>(ops[index], std::move(left), std::move(right), symbols[index]);
    }
}


Tree tournamentSelection(const std::vector<Tree>& population, int tournamentSize, std::map<std::string, double>& vars, double targetValue, std::mt19937& gen) {
    std::uniform_int_distribution<> dis(0, population.size() - 1);
    double bestFitness = std::numeric_limits<double>::infinity();
    int bestIndex = 0;

    for (int i = 0; i < tournamentSize; ++i) {
        int idx = dis(gen);
        double currentFitness = population[idx].fitness(vars, targetValue);
        if (currentFitness < bestFitness) {
            bestFitness = currentFitness;
            bestIndex = idx;
        }
    }

    return Tree(population[bestIndex].root->clone());
}

std::unique_ptr<Node> generateSubtree(int maxHeight, std::mt19937& gen) {
    if (maxHeight <= 1) {
        if (std::uniform_int_distribution<>(0, 1)(gen) == 0) {
            return std::make_unique<TerminalNode>(terminals[std::uniform_int_distribution<>(0, terminals.size() - 1)(gen)]);
        } else {
            return std::make_unique<ConstantNode>(std::uniform_real_distribution<>(0, 100)(gen));
        }
    }
    int index = std::uniform_int_distribution<>(0, ops.size() - 1)(gen);
    auto left = generateSubtree(maxHeight - 1, gen);
    auto right = generateSubtree(maxHeight - 1, gen);
    return std::make_unique<OperatorNode>(ops[index], std::move(left), std::move(right), symbols[index]);
}


int treeHeight(const std::unique_ptr<Node>& node) {
    if (!node) return 0;
    if (auto opNode = dynamic_cast<OperatorNode*>(node.get())) {
        return 1 + std::max(treeHeight(opNode->left), treeHeight(opNode->right));
    }
    return 1;
}


void mutateSubtree(std::unique_ptr<Node>& node, int depth, int targetDepth, std::mt19937& gen, int& didMutate) {
    if (!node || didMutate) return;

    if (depth == targetDepth) {
        int maxHeight = std::max(1, 4 - depth);
        node = generateSubtree(maxHeight, gen);
        didMutate = true;
        return;
    }

    if (auto opNode = dynamic_cast<OperatorNode*>(node.get())) {
        mutateSubtree(opNode->left, depth + 1, targetDepth, gen, didMutate);
        mutateSubtree(opNode->right, depth + 1, targetDepth, gen, didMutate);
    }
}

void mutateTree(Tree& tree, std::mt19937& gen) {
    int height = treeHeight(tree.root);
    if (height == 1) {
        tree.root = generateSubtree(1, gen);
    } else {
        int targetDepth = std::uniform_int_distribution<>(2, height)(gen);
        int didMutate = false;
        mutateSubtree(tree.root, 1, targetDepth, gen, didMutate);
    }
}

void onePointCrossover(Tree& tree1, Tree& tree2, std::mt19937& rng) {
    auto nodes1 = tree1.getAllNodes(); // Gets pointers to all unique_ptr<Node> in tree1
    auto nodes2 = tree2.getAllNodes(); // Gets pointers to all unique_ptr<Node> in tree2

    // Ensure there are nodes to perform crossover in both trees
    if (nodes1.size() < 2 || nodes2.size() < 2) return; // No crossover if only root node is present

    // Select random non-root nodes from each tree for crossover
    std::uniform_int_distribution<> dist1(1, nodes1.size() - 1); // Exclude root
    std::uniform_int_distribution<> dist2(1, nodes2.size() - 1); // Exclude root

    int index1 = dist1(rng);
    int index2 = dist2(rng);

    // Directly swap the subtrees between tree1 and tree2
    std::swap(*nodes1[index1], *nodes2[index2]);
}

constexpr double mutationRate = 0.1;
constexpr double crossoverRate = 0.9;
constexpr int populationSize = 200;
constexpr int generations = 40;
constexpr double targetValue = 50.0;

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> probDist(0.0, 1.0);

    std::unique_ptr<Tree> bestOverall = nullptr;
    double bestOverallFitness = std::numeric_limits<double>::infinity();

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<Tree> population;
        for (int i = 0; i < populationSize; ++i) {
            population.emplace_back(createRandomTree(6, rng));
        }

        for (int i = 0; i < populationSize; i += 2) {
            if (probDist(rng) < crossoverRate && i + 1 < populationSize) {
                std::cout << population[i].toString();
                std::cout << "\n\n";
                std::cout << population[i + 1].toString();
                onePointCrossover(population[i], population[i+1], rng);
                std::cout << population[i].toString();

            }
        }

        for (Tree& tree : population) {
            if (probDist(rng) < mutationRate) {
                mutateTree(tree, rng);
            }
        }

        Tree* bestInGeneration = &population[0];
        double bestFitnessInGeneration = bestInGeneration->fitness(vars, targetValue);
        for (Tree& tree : population) {
            double currentFitness = tree.fitness(vars, targetValue);
            if (currentFitness < bestFitnessInGeneration) {
                bestInGeneration = &tree;
                bestFitnessInGeneration = currentFitness;
            }

            if (currentFitness < bestOverallFitness) {
                bestOverallFitness = currentFitness;
                bestOverall = std::make_unique<Tree>(tree.root->clone());
            }
        }

        std::cout << "Generation " << gen << " Best Individual:\n";
        std::cout << "Function: " << bestInGeneration->toString() << "\n";
        std::cout << "Fitness: " << bestFitnessInGeneration << "\n";
        std::cout << "Evaluation: " << bestInGeneration->evaluate(vars) << "\n\n";
    }

    if (bestOverall) {
        std::cout << "Best Overall Individual:\n";
        std::cout << "Function: " << bestOverall->toString() << "\n";
        std::cout << "Fitness: " << bestOverallFitness << "\n";
        std::cout << "Evaluation: " << bestOverall->evaluate(vars) << "\n";
    } else {
        std::cout << "No overall best individual found.\n";
    }

    return 0;
}
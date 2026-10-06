#include "Parser.hpp"
#include "Config.hpp"

#include <cctype>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>

namespace quarz {

namespace {

using Op = ParserExec::Op;

struct Node {
    enum Kind { NUM, VAR, OP } kind = NUM;
    double val = 0;
    int var = 0;
    Op op = Op::ADD;
    std::vector<std::unique_ptr<Node>> kids;
};
using NodeP = std::unique_ptr<Node>;

NodeP num(double v) { auto n = std::make_unique<Node>(); n->kind = Node::NUM; n->val = v; return n; }
NodeP mk(Op op, std::vector<NodeP> kids) {
    auto n = std::make_unique<Node>();
    n->kind = Node::OP; n->op = op; n->kids = std::move(kids);
    return n;
}

struct FuncInfo { Op op; int nargs; };
const std::map<std::string, FuncInfo>& functions() {
    static const std::map<std::string, FuncInfo> f = {
        {"sqrt", {Op::SQRT, 1}}, {"exp", {Op::EXP, 1}}, {"log", {Op::LOG, 1}}, {"log10", {Op::LOG10, 1}},
        {"sin", {Op::SIN, 1}}, {"cos", {Op::COS, 1}}, {"tan", {Op::TAN, 1}}, {"asin", {Op::ASIN, 1}},
        {"acos", {Op::ACOS, 1}}, {"atan", {Op::ATAN, 1}}, {"sinh", {Op::SINH, 1}}, {"cosh", {Op::COSH, 1}},
        {"tanh", {Op::TANH, 1}}, {"abs", {Op::ABS, 1}}, {"fabs", {Op::ABS, 1}}, {"floor", {Op::FLOOR, 1}},
        {"ceil", {Op::CEIL, 1}}, {"erf", {Op::ERF, 1}}, {"heaviside", {Op::STEP, 1}}, {"step", {Op::STEP, 1}},
        {"pow", {Op::POW, 2}}, {"atan2", {Op::ATAN2, 2}}, {"min", {Op::MIN, 2}}, {"max", {Op::MAX, 2}},
        {"mod", {Op::MOD, 2}}, {"fmod", {Op::MOD, 2}}, {"if", {Op::IF, 3}}};
    return f;
}

class Compiler {
public:
    Compiler(const std::string& s, const std::vector<std::string>& vars, const std::map<std::string, double>& consts)
        : s_(s), vars_(vars), consts_(consts) {}

    NodeP parse() {
        NodeP n = parse_or();
        skip();
        if (p_ != s_.size()) fail("unexpected '" + std::string(1, s_[p_]) + "'");
        return n;
    }

private:
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("parser: " + msg + " at position " + std::to_string(p_ + 1) + " in \"" + s_ + "\"");
    }
    void skip() { while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_; }
    bool eat(const char* tok) {
        skip();
        const size_t n = std::char_traits<char>::length(tok);
        if (s_.compare(p_, n, tok) == 0) { p_ += n; return true; }
        return false;
    }

    NodeP parse_or() {
        NodeP a = parse_and();
        while (eat("||")) { std::vector<NodeP> k; k.push_back(std::move(a)); k.push_back(parse_and()); a = mk(Op::OR, std::move(k)); }
        return a;
    }
    NodeP parse_and() {
        NodeP a = parse_cmp();
        while (eat("&&")) { std::vector<NodeP> k; k.push_back(std::move(a)); k.push_back(parse_cmp()); a = mk(Op::AND, std::move(k)); }
        return a;
    }
    NodeP parse_cmp() {
        NodeP a = parse_add();
        for (;;) {
            Op op;
            if (eat("==")) op = Op::EQ;
            else if (eat("!=")) op = Op::NE;
            else if (eat("<=")) op = Op::LE;
            else if (eat(">=")) op = Op::GE;
            else if (eat("<")) op = Op::LT;
            else if (eat(">")) op = Op::GT;
            else return a;
            std::vector<NodeP> k; k.push_back(std::move(a)); k.push_back(parse_add());
            a = mk(op, std::move(k));
        }
    }
    NodeP parse_add() {
        NodeP a = parse_mul();
        for (;;) {
            Op op;
            if (eat("+")) op = Op::ADD;
            else if (eat("-")) op = Op::SUB;
            else return a;
            std::vector<NodeP> k; k.push_back(std::move(a)); k.push_back(parse_mul());
            a = mk(op, std::move(k));
        }
    }
    NodeP parse_mul() {
        NodeP a = parse_unary();
        for (;;) {
            Op op;
            skip();
            if (s_.compare(p_, 2, "**") == 0) return a;   // power, handled below
            if (eat("*")) op = Op::MUL;
            else if (eat("/")) op = Op::DIV;
            else return a;
            std::vector<NodeP> k; k.push_back(std::move(a)); k.push_back(parse_unary());
            a = mk(op, std::move(k));
        }
    }
    NodeP parse_unary() {
        if (eat("-")) { std::vector<NodeP> k; k.push_back(parse_unary()); return mk(Op::NEG, std::move(k)); }
        if (eat("+")) return parse_unary();
        skip();
        if (p_ < s_.size() && s_[p_] == '!' && (p_ + 1 >= s_.size() || s_[p_ + 1] != '=')) {
            ++p_;
            std::vector<NodeP> k; k.push_back(parse_unary()); return mk(Op::NOT, std::move(k));
        }
        return parse_pow();
    }
    NodeP parse_pow() {
        NodeP a = parse_primary();
        if (eat("^") || eat("**")) {   // right associative; exponent may carry a sign
            std::vector<NodeP> k; k.push_back(std::move(a)); k.push_back(parse_unary());
            return mk(Op::POW, std::move(k));
        }
        return a;
    }
    NodeP parse_primary() {
        skip();
        if (p_ >= s_.size()) fail("unexpected end of expression");
        const char c = s_[p_];
        if (c == '(') {
            ++p_;
            NodeP a = parse_or();
            if (!eat(")")) fail("missing ')'");
            return a;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const char* b = s_.c_str() + p_;
            char* e = nullptr;
            const double v = std::strtod(b, &e);
            if (e == b) fail("bad number");
            p_ += static_cast<size_t>(e - b);
            return num(v);
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const size_t b = p_;
            while (p_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[p_])) || s_[p_] == '_')) ++p_;
            const std::string id = s_.substr(b, p_ - b);
            skip();
            if (p_ < s_.size() && s_[p_] == '(') {   // function call
                auto it = functions().find(id);
                if (it == functions().end()) fail("unknown function '" + id + "'");
                ++p_;
                std::vector<NodeP> args;
                if (!eat(")")) {
                    for (;;) {
                        args.push_back(parse_or());
                        if (eat(")")) break;
                        if (!eat(",")) fail("expected ',' or ')' in call of " + id);
                    }
                }
                if (static_cast<int>(args.size()) != it->second.nargs)
                    fail(id + "() needs " + std::to_string(it->second.nargs) + " argument(s)");
                return mk(it->second.op, std::move(args));
            }
            for (size_t i = 0; i < vars_.size(); ++i)
                if (vars_[i] == id) { auto n = std::make_unique<Node>(); n->kind = Node::VAR; n->var = static_cast<int>(i); return n; }
            auto ct = consts_.find(id);
            if (ct != consts_.end()) return num(ct->second);
            if (id == "pi") return num(PI);
            if (id == "e") return num(std::exp(1.0));
            fail("unknown symbol '" + id + "' (not a variable, my_constants entry or function)");
        }
        fail("unexpected '" + std::string(1, c) + "'");
    }

    const std::string& s_;
    const std::vector<std::string>& vars_;
    const std::map<std::string, double>& consts_;
    size_t p_ = 0;
};

// constant folding: evaluate every subtree without variables once
bool fold(NodeP& n) {
    if (n->kind == Node::NUM) return true;
    if (n->kind == Node::VAR) return false;
    bool all = true;
    for (auto& k : n->kids) all = fold(k) && all;
    if (!all) return false;
    ParserExec e;
    for (auto& k : n->kids) { e.cval[e.n] = k->val; e.op[e.n] = Op::PUSHC; e.arg[e.n] = static_cast<unsigned char>(e.n); ++e.n; }
    e.op[e.n++] = n->op;
    // fix const indices for the PUSHC we used
    const double v = e.eval(nullptr);
    n = num(v);
    return true;
}

void emit(const Node& n, ParserExec& e, int& nconst, int depth, int& maxdepth) {
    auto push = [&](Op op, int arg) {
        if (e.n >= ParserExec::MAXI) throw std::runtime_error("parser: expression too long");
        e.op[e.n] = op;
        e.arg[e.n] = static_cast<unsigned char>(arg);
        ++e.n;
    };
    if (n.kind == Node::NUM) {
        if (nconst >= ParserExec::MAXC) throw std::runtime_error("parser: too many constants in expression");
        e.cval[nconst] = n.val;
        push(Op::PUSHC, nconst++);
        maxdepth = std::max(maxdepth, depth + 1);
        return;
    }
    if (n.kind == Node::VAR) { push(Op::PUSHV, n.var); maxdepth = std::max(maxdepth, depth + 1); return; }
    int d = depth;
    for (const auto& k : n.kids) { emit(*k, e, nconst, d, maxdepth); ++d; }
    push(n.op, 0);
}

std::string strip_quotes(std::string v) {
    while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.pop_back();
    size_t b = 0;
    while (b < v.size() && std::isspace(static_cast<unsigned char>(v[b]))) ++b;
    v = v.substr(b);
    if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\'')))
        v = v.substr(1, v.size() - 2);
    return v;
}

} // namespace

Parser::Parser(const std::string& expr, const std::vector<std::string>& vars,
               const std::map<std::string, double>& consts)
    : expr_(strip_quotes(expr)) {
    if (static_cast<int>(vars.size()) > ParserExec::MAXV) throw std::runtime_error("parser: at most 4 variables");
    Compiler c(expr_, vars, consts);
    NodeP root = c.parse();
    constant_ = fold(root);
    int nconst = 0, maxdepth = 0;
    emit(*root, exec_, nconst, 0, maxdepth);
    if (maxdepth > ParserExec::MAXS) throw std::runtime_error("parser: expression nested too deeply: " + expr_);
    exec_.nvars = static_cast<int>(vars.size());
}

std::map<std::string, double> read_constants(const Config& cfg) {
    std::map<std::string, std::string> raw;
    const std::string pre = "my_constants.";
    for (const auto& k : cfg.keys())
        if (k.compare(0, pre.size(), pre) == 0) raw[k.substr(pre.size())] = cfg.get_string(k);
    std::map<std::string, double> out;
    std::set<std::string> busy;
    std::function<void(const std::string&)> resolve = [&](const std::string& name) {
        if (out.count(name)) return;
        if (busy.count(name)) throw std::runtime_error("my_constants: circular definition involving '" + name + "'");
        busy.insert(name);
        // resolve every other constant this one may refer to, then compile
        for (;;) {
            try {
                out[name] = Parser(raw.at(name), {}, out)();
                break;
            } catch (const std::runtime_error& e) {
                const std::string msg = e.what();
                const std::string tag = "unknown symbol '";
                const auto p = msg.find(tag);
                if (p == std::string::npos) throw std::runtime_error("my_constants." + name + ": " + msg);
                const std::string dep = msg.substr(p + tag.size(), msg.find('\'', p + tag.size()) - p - tag.size());
                if (!raw.count(dep)) throw std::runtime_error("my_constants." + name + ": " + msg);
                resolve(dep);
            }
        }
        busy.erase(name);
    };
    for (const auto& kv : raw) resolve(kv.first);
    return out;
}

bool read_function(const Config& cfg, const std::string& name, int nvars, Parser& p,
                   const std::vector<std::string>& default_vars, bool allow_plain) {
    const auto consts = read_constants(cfg);
    for (const auto& k : cfg.keys()) {
        if (k.size() <= name.size() || k.compare(0, name.size(), name) != 0 || k[name.size()] != '(') continue;
        if (k.back() != ')') throw std::runtime_error("input: malformed function key '" + k + "'");
        std::vector<std::string> vars;
        std::string cur;
        for (size_t i = name.size() + 1; i + 1 < k.size(); ++i) {
            const char c = k[i];
            if (c == ',') { vars.push_back(cur); cur.clear(); }
            else if (!std::isspace(static_cast<unsigned char>(c))) cur += c;
        }
        if (!cur.empty()) vars.push_back(cur);
        if (static_cast<int>(vars.size()) != nvars)
            throw std::runtime_error("input: '" + k + "' must have " + std::to_string(nvars) + " variables");
        p = Parser(cfg.get_string(k), vars, consts);
        return true;
    }
    if (allow_plain && cfg.has(name)) {
        p = Parser(cfg.get_string(name), default_vars, consts);
        return true;
    }
    return false;
}

} // namespace quarz

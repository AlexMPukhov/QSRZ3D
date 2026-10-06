// Math expression parser for input-file functions (in the spirit of the AMReX
// parser used by HiPACE++ / WarpX).
//
//   my_constants.kp  = 1.0
//   my_constants.L   = 2*pi/kp            (constants may use earlier constants)
//   plasma.density(x,y,z) = "if(z < L, z/L, 1) * (1 + 0.3*(x^2+y^2))"
//
// An expression is compiled once on the host to a small stack program
// (ParserExec).  ParserExec is trivially copyable and can be evaluated on the
// host and inside Kokkos kernels (GPU).
//
// Grammar (lowest to highest precedence)
//   a || b     a && b     == != < > <= >=     + -     * /     unary - + !     ^ (right assoc., also **)
// Numbers: 1, 2.5, .5, 1e-3.  Built-in constants: pi, e (unless shadowed).
// Functions:
//   one argument : sqrt exp log log10 sin cos tan asin acos atan sinh cosh tanh
//                  abs floor ceil erf heaviside(x)=step(x) (1 for x>0, 0 for x<0, 1/2 at 0)
//   two arguments: pow(a,b) atan2(y,x) min(a,b) max(a,b) mod(a,b) (fmod)
//   three        : if(cond, a, b)    (cond != 0 -> a, else b)
#pragma once

#include "Types.hpp"

#include <map>
#include <string>
#include <vector>

namespace quarz {

class Config;

struct ParserExec {
    static constexpr int MAXI = 256;   // instructions
    static constexpr int MAXC = 64;    // constants
    static constexpr int MAXS = 32;    // evaluation stack
    static constexpr int MAXV = 4;     // variables

    enum Op : unsigned char {
        PUSHC, PUSHV, ADD, SUB, MUL, DIV, POW, NEG, NOT, LT, GT, LE, GE, EQ, NE, AND, OR,
        SQRT, EXP, LOG, LOG10, SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, ABS, FLOOR, CEIL, ERF, STEP,
        ATAN2, MIN, MAX, MOD, IF
    };

    int n = 0;
    int nvars = 0;
    unsigned char op[MAXI] = {};
    unsigned char arg[MAXI] = {};
    double cval[MAXC] = {};

    KOKKOS_INLINE_FUNCTION double eval(const double* v) const {
        double s[MAXS];
        int sp = 0;
        for (int i = 0; i < n; ++i) {
            switch (op[i]) {
                case PUSHC: s[sp++] = cval[arg[i]]; break;
                case PUSHV: s[sp++] = v[arg[i]]; break;
                case ADD: --sp; s[sp - 1] += s[sp]; break;
                case SUB: --sp; s[sp - 1] -= s[sp]; break;
                case MUL: --sp; s[sp - 1] *= s[sp]; break;
                case DIV: --sp; s[sp - 1] /= s[sp]; break;
                case POW: --sp; s[sp - 1] = ipow(s[sp - 1], s[sp]); break;
                case NEG: s[sp - 1] = -s[sp - 1]; break;
                case NOT: s[sp - 1] = (s[sp - 1] == 0.0) ? 1.0 : 0.0; break;
                case LT: --sp; s[sp - 1] = s[sp - 1] < s[sp] ? 1.0 : 0.0; break;
                case GT: --sp; s[sp - 1] = s[sp - 1] > s[sp] ? 1.0 : 0.0; break;
                case LE: --sp; s[sp - 1] = s[sp - 1] <= s[sp] ? 1.0 : 0.0; break;
                case GE: --sp; s[sp - 1] = s[sp - 1] >= s[sp] ? 1.0 : 0.0; break;
                case EQ: --sp; s[sp - 1] = s[sp - 1] == s[sp] ? 1.0 : 0.0; break;
                case NE: --sp; s[sp - 1] = s[sp - 1] != s[sp] ? 1.0 : 0.0; break;
                case AND: --sp; s[sp - 1] = (s[sp - 1] != 0.0 && s[sp] != 0.0) ? 1.0 : 0.0; break;
                case OR: --sp; s[sp - 1] = (s[sp - 1] != 0.0 || s[sp] != 0.0) ? 1.0 : 0.0; break;
                case SQRT: s[sp - 1] = Kokkos::sqrt(s[sp - 1]); break;
                case EXP: s[sp - 1] = Kokkos::exp(s[sp - 1]); break;
                case LOG: s[sp - 1] = Kokkos::log(s[sp - 1]); break;
                case LOG10: s[sp - 1] = Kokkos::log10(s[sp - 1]); break;
                case SIN: s[sp - 1] = Kokkos::sin(s[sp - 1]); break;
                case COS: s[sp - 1] = Kokkos::cos(s[sp - 1]); break;
                case TAN: s[sp - 1] = Kokkos::tan(s[sp - 1]); break;
                case ASIN: s[sp - 1] = Kokkos::asin(s[sp - 1]); break;
                case ACOS: s[sp - 1] = Kokkos::acos(s[sp - 1]); break;
                case ATAN: s[sp - 1] = Kokkos::atan(s[sp - 1]); break;
                case SINH: s[sp - 1] = Kokkos::sinh(s[sp - 1]); break;
                case COSH: s[sp - 1] = Kokkos::cosh(s[sp - 1]); break;
                case TANH: s[sp - 1] = Kokkos::tanh(s[sp - 1]); break;
                case ABS: s[sp - 1] = Kokkos::fabs(s[sp - 1]); break;
                case FLOOR: s[sp - 1] = Kokkos::floor(s[sp - 1]); break;
                case CEIL: s[sp - 1] = Kokkos::ceil(s[sp - 1]); break;
                case ERF: s[sp - 1] = Kokkos::erf(s[sp - 1]); break;
                case STEP: s[sp - 1] = s[sp - 1] > 0.0 ? 1.0 : (s[sp - 1] < 0.0 ? 0.0 : 0.5); break;
                case ATAN2: --sp; s[sp - 1] = Kokkos::atan2(s[sp - 1], s[sp]); break;
                case MIN: --sp; s[sp - 1] = s[sp - 1] < s[sp] ? s[sp - 1] : s[sp]; break;
                case MAX: --sp; s[sp - 1] = s[sp - 1] > s[sp] ? s[sp - 1] : s[sp]; break;
                case MOD: --sp; s[sp - 1] = Kokkos::fmod(s[sp - 1], s[sp]); break;
                case IF: sp -= 2; s[sp - 1] = (s[sp - 1] != 0.0) ? s[sp] : s[sp + 1]; break;
                default: break;
            }
        }
        return sp > 0 ? s[0] : 0.0;
    }
    KOKKOS_INLINE_FUNCTION double operator()(double a = 0, double b = 0, double c = 0, double d = 0) const {
        const double v[MAXV] = {a, b, c, d};
        return eval(v);
    }

    // integer powers by multiplication (exact, fast, safe for negative bases)
    KOKKOS_INLINE_FUNCTION static double ipow(double a, double b) {
        if (b == Kokkos::floor(b) && Kokkos::fabs(b) <= 16.0) {
            int k = static_cast<int>(b);
            const bool inv = k < 0;
            if (inv) k = -k;
            double r = 1.0, x = a;
            while (k) { if (k & 1) r *= x; x *= x; k >>= 1; }
            return inv ? 1.0 / r : r;
        }
        return Kokkos::pow(a, b);
    }
};

class Parser {
public:
    Parser() = default;
    // vars: variable names in argument order; consts: user constants (my_constants)
    Parser(const std::string& expr, const std::vector<std::string>& vars,
           const std::map<std::string, double>& consts = {});

    const ParserExec& exec() const { return exec_; }
    double operator()(double a = 0, double b = 0, double c = 0, double d = 0) const { return exec_(a, b, c, d); }
    bool is_constant() const { return constant_; }   // expression does not depend on the variables
    const std::string& expression() const { return expr_; }

private:
    std::string expr_;
    ParserExec exec_;
    bool constant_ = false;
};

// my_constants.<name> = <expression of earlier/other constants>; resolved with cycle detection
std::map<std::string, double> read_constants(const Config& cfg);

// Function given in the input as   <name>(v1,v2,...) = "expression".
// Returns false if no such key exists.  The variable names are taken from the key,
// their number must equal nvars.  If allow_plain, a plain  <name> = expression  (using the
// variable names default_vars) is accepted as well.
bool read_function(const Config& cfg, const std::string& name, int nvars, Parser& p,
                   const std::vector<std::string>& default_vars, bool allow_plain = false);

} // namespace quarz

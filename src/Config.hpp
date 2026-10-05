// Minimal "key = value" input-file parser.
//   - '#' starts a comment
//   - values are whitespace separated lists; get_list() returns all tokens
//   - unused keys are reported at the end of the run (catches typos)
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>

namespace qsrz {

class Config {
public:
    Config() = default;
    explicit Config(const std::string& filename);

    void set(const std::string& key, const std::string& value) { kv_[key] = value; }
    bool has(const std::string& key) const { return kv_.count(key) > 0; }

    double      get_double(const std::string& key) const;                 // required
    double      get_double(const std::string& key, double def) const;     // optional
    int         get_int(const std::string& key) const;
    int         get_int(const std::string& key, int def) const;
    bool        get_bool(const std::string& key, bool def) const;
    std::string get_string(const std::string& key) const;
    std::string get_string(const std::string& key, const std::string& def) const;
    std::vector<std::string> get_list(const std::string& key) const;       // empty if missing
    // list of "a:b" pairs -> vector of (a,b)
    std::vector<std::pair<double, double>> get_pairs(const std::string& key) const;

    std::vector<std::string> unused_keys() const;
    std::vector<std::string> keys() const;
    void print(std::ostream& os) const;

private:
    const std::string& raw(const std::string& key) const;
    std::map<std::string, std::string> kv_;
    mutable std::set<std::string> used_;
};

} // namespace qsrz

#include "Config.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace qsrz {

static std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

Config::Config(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Config: cannot open input file '" + filename + "'");
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("Config: line " + std::to_string(lineno) + ": missing '='");
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty())
            throw std::runtime_error("Config: line " + std::to_string(lineno) + ": empty key");
        kv_[key] = val;
    }
}

const std::string& Config::raw(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("Config: required key '" + key + "' is missing");
    used_.insert(key);
    return it->second;
}

double Config::get_double(const std::string& key) const {
    const std::string& v = raw(key);
    try { return std::stod(v); }
    catch (...) { throw std::runtime_error("Config: key '" + key + "' is not a number: " + v); }
}
double Config::get_double(const std::string& key, double def) const {
    return has(key) ? get_double(key) : def;
}
int Config::get_int(const std::string& key) const {
    const std::string& v = raw(key);
    try { return static_cast<int>(std::stol(v)); }
    catch (...) { throw std::runtime_error("Config: key '" + key + "' is not an integer: " + v); }
}
int Config::get_int(const std::string& key, int def) const {
    return has(key) ? get_int(key) : def;
}
bool Config::get_bool(const std::string& key, bool def) const {
    if (!has(key)) return def;
    std::string v = raw(key);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::runtime_error("Config: key '" + key + "' is not a boolean: " + v);
}
std::string Config::get_string(const std::string& key) const { return raw(key); }
std::string Config::get_string(const std::string& key, const std::string& def) const {
    return has(key) ? raw(key) : def;
}
std::vector<std::string> Config::get_list(const std::string& key) const {
    std::vector<std::string> out;
    if (!has(key)) return out;
    std::istringstream ss(raw(key));
    std::string tok;
    while (ss >> tok) out.push_back(tok);
    return out;
}
std::vector<std::pair<double, double>> Config::get_pairs(const std::string& key) const {
    std::vector<std::pair<double, double>> out;
    for (const auto& tok : get_list(key)) {
        auto c = tok.find(':');
        if (c == std::string::npos)
            throw std::runtime_error("Config: key '" + key + "': expected a:b pairs, got '" + tok + "'");
        out.emplace_back(std::stod(tok.substr(0, c)), std::stod(tok.substr(c + 1)));
    }
    return out;
}

std::vector<std::string> Config::keys() const {
    std::vector<std::string> out;
    for (const auto& kv : kv_) out.push_back(kv.first);
    return out;
}

std::vector<std::string> Config::unused_keys() const {
    std::vector<std::string> out;
    for (const auto& kv : kv_)
        if (!used_.count(kv.first)) out.push_back(kv.first);
    return out;
}

void Config::print(std::ostream& os) const {
    for (const auto& kv : kv_) os << "  " << kv.first << " = " << kv.second << "\n";
}

} // namespace qsrz

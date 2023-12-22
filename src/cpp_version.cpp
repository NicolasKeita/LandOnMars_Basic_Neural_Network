#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

class RocketLandingEnv {
private:
    std::vector<std::pair<int, int>> surface;
    std::vector<std::pair<int, int>> surface_segments;
    std::vector<std::pair<int, int>> landing_spot;
    std::pair<int, int> middle_landing_spot;
    std::vector<std::pair<int, int>> path_to_the_landing_spot;
    std::vector<float> rewards_episode;
    std::vector<float> reward_plot;
    std::vector<std::pair<int, int>> trajectory_plot;
    std::vector<float> initial_state;
    std::vector<float> state;
    std::vector<int> action_constraints;
    float gravity;
    int n_intermediate_path;
    float initial_fuel;
    std::vector<float> prev_shaping;
    int i_intermediate_path;

public:
    RocketLandingEnv(std::vector<int> initial_state, std::vector<std::pair<int, int>> surface) {
        this->n_intermediate_path = 6;
        std::vector<int> initial_pos = {initial_state[0], initial_state[1]};
        int initial_hs = initial_state[2];
        int initial_vs = initial_state[3];
        int initial_rotation = initial_state[5];
        int initial_thrust = initial_state[6];
        this->initial_fuel = initial_state[4];
        this->rewards_episode = {};
        this->reward_plot = {};
        this->trajectory_plot = {};
        this->surface = surface;
        for (size_t i = 0; i < surface.size() - 1; ++i) {
            surface_segments.push_back(std::make_pair(surface[i].second, surface[i + 1].first));
        }
        this->landing_spot = this->find_landing_spot(this->surface);
        this->middle_landing_spot = {(this->landing_spot[0].first + this->landing_spot[1].first) / 2,
                                     (this->landing_spot[0].second + this->landing_spot[1].second) / 2};
        this->path_to_the_landing_spot = this->search_path(initial_pos, this->surface, this->landing_spot, {});
        std::vector<std::pair<int, int>> temp_path_to_the_landing_spot;
        for (int i = 0; i < this->path_to_the_landing_spot.size() - 1; i++) {
            if (i < this->path_to_the_landing_spot.size() - 1) {
                temp_path_to_the_landing_spot.push_back({this->path_to_the_landing_spot[i].first,
                                                         this->path_to_the_landing_spot[i].second + 200});
            } else {
                temp_path_to_the_landing_spot.push_back(this->path_to_the_landing_spot[i]);
            }
        }
        this->path_to_the_landing_spot = temp_path_to_the_landing_spot;
        this->initial_state = {
                static_cast<float>(initial_pos[0]),
                static_cast<float>(initial_pos[1]),
                static_cast<float>(initial_hs),
                static_cast<float>(initial_vs),
                static_cast<float>(this->initial_fuel),
                static_cast<float>(initial_rotation),
                static_cast<float>(initial_thrust),
                static_cast<float>(this->distance_to_line(initial_pos[0], initial_pos[1], {this->landing_spot})),
                static_cast<float>(this->distance_to_line(initial_pos[0], initial_pos[1], this->surface_segments)),
                0.0
        };
        this->state = this->initial_state;
        this->action_constraints = {15, 1};
        this->gravity = 3.711;
    }

    std::vector<std::pair<int, int>> find_landing_spot(std::vector<std::pair<int, int>> planet_surface) {
        for (int i = 0; i < planet_surface.size() - 1; i++) {
            if (planet_surface[i].second == planet_surface[i + 1].second) {
                return {planet_surface[i], planet_surface[i + 1]};
            }
        }
        throw std::runtime_error("No landing site on test-case data");
    }

    std::vector<std::pair<int, int>> search_path(std::vector<int> initial_pos,
                                                 std::vector<std::pair<int, int>> surface,
                                                 std::vector<std::pair<int, int>> landing_spot,
                                                 std::vector<std::pair<int, int>> my_path) {
        std::vector<std::pair<int, int>> path;
        for (auto s2 : this->surface_segments) {
            if (this->do_segments_intersect({initial_pos, this->middle_landing_spot}, s2)) {
                path = {s2[0] if s2[0].second > s2[1].second else s2[1]};
                break;
            }
        }
        if (path.empty()) {
            std::vector<std::pair<int, int>> t;
            for (int i = 0; i < this->n_intermediate_path; i++) {
                t.push_back(initial_pos + (this->middle_landing_spot - initial_pos) * i / (this->n_intermediate_path - 1));
            }
            return my_path.empty() ? t : my_path[:-1] + t;
        }
        path[1] = path[1];
        return this->search_path(path, surface, landing_spot,
                                 initial_pos + (path - initial_pos) * i / (this->n_intermediate_path - 1));
    }

    float distance_2(std::pair<int, int> a, std::pair<int, int> b) {
        return std::pow(a.first - b.first, 2) + std::pow(a.second - b.second, 2);
    }

    float distance_to_line(int x, int y, std::vector<std::pair<int, int>> line_segments) {
        float min_distance_squared = std::numeric_limits<float>::infinity();
        for (auto segment : line_segments) {
            std::pair<int, int> p1 = segment[0];
            std::pair<int, int> p2 = segment[1];
            int x1 = p1.first;
            int y1 = p1.second;
            int x2 = p2.first;
            int y2 = p2.second;
            int dx = x2 - x1;
            int dy = y2 - y1;
            int dot_product = (x - x1) * dx + (y - y1) * dy;
            float t = std::max(0.0f, std::min(1.0f, static_cast<float>(dot_product) / (dx * dx + dy * dy)));
            std::pair<int, int> closest_point = {x1 + t * dx, y1 + t * dy};
            float segment_distance_squared = this->distance_2({x, y}, closest_point);
            min_distance_squared = std::min(min_distance_squared, segment_distance_squared);
        }
        return min_distance_squared;
    }

    std::tuple<double, bool, bool> computeReward(const std::vector<double>& state) {
        double x, y, hs, vs, remainingFuel, rotation, distLandingSpot, distSurface, distPath;
        std::tie(x, y, hs, vs, remainingFuel, rotation, std::ignore, distLandingSpot, distSurface, distPath) = state;

        bool isSuccessfulLanding = distLandingSpot < 1 && rotation == 0 && std::abs(vs) <= 40 && std::abs(hs) <= 20;
        bool isCrashedOnLandingSpot = distLandingSpot < 1;
        bool isCrashedAnywhere = y <= 1 || y >= 3000 - 1 || x <= 1 || x >= 7000 - 1 || distSurface < 1 || remainingFuel < 4;
        bool isCloseToLand = distLandingSpot < 1500 * 1500;

        distPath = isCloseToLand ? distLandingSpot : distPath;

        double reward = (norm_reward(distPath, 0, 7500 * 7500)
                        + 0.65 * norm_reward(std::abs(vs), 39, 140)
                        + 0.35 * norm_reward(std::abs(hs), 19, 140)
        );

        if (isSuccessfulLanding) {
            return {1000 + remainingFuel * 100, true, false};
        } else if (isCrashedOnLandingSpot) {
            return {reward - 100, true, false};
        } else if (isCrashedAnywhere) {
            return {reward - 100, false, true};
        } else {
            return {reward, false, false};
        }
    }

    std::vector<float> _compute_next_state(std::vector<float> state, std::vector<int> action) {
        float x = state[0];
        float y = state[1];
        float hs = state[2];
        float vs = state[3];
        float remaining_fuel = state[4];
        float rotation = state[5];
        float thrust = state[6];
        float dist_landing_spot = state[7];
        float dist_surface = state[8];
        float dist_path = state[9];
        std::vector<int> rotation_limits = this->_generate_action_limits(rotation, 15, -90, 90);
        std::vector<int> thrust_limits = this->_generate_action_limits(thrust, 1, 0, 4);
        int new_rotation = std::clamp(action[0], rotation_limits[0], rotation_limits[1]);
        int new_thrust = std::clamp(action[1], thrust_limits[0], thrust_limits[1]);
        float radians = M_PI / 180 * new_rotation;
        float x_acceleration = std::sin(radians) * new_thrust;
        float y_acceleration = std::cos(radians) * new_thrust - this->gravity;
        float new_horizontal_speed = hs - x_acceleration;
        float new_vertical_speed = vs + y_acceleration;
        float new_x = x + hs - 0.5 * x_acceleration;
        float new_y = y + vs + 0.5 * y_acceleration;
        std::pair<float, float> new_pos = {std::clamp(new_x, 0.0f, 7000.0f), std::clamp(new_y, 0.0f, 3000.0f)};
        new_pos = this->calculate_intersection({x, y}, new_pos, this->surface);
        remaining_fuel = std::max(remaining_fuel - new_thrust, 0.0f);
        std::vector<std::pair<int, int>> surface_segments;
        for (int i = 0; i < this->surface.size() - 1; i++) {
            surface_segments.push_back({this->surface[i], this->surface[i + 1]});
        }
        float dist_to_landing_spot = this->distance_to_line(new_pos.first, new_pos.second, {this->landing_spot});
        float dist_to_surface = this->distance_to_line(new_pos.first, new_pos.second, surface_segments);
        float dist_to_path = this->get_distance_to_path(new_pos, this->path_to_the_landing_spot);
        std::vector<float> new_state = {
                new_pos.first,
                new_pos.second,
                new_horizontal_speed,
                new_vertical_speed,
                remaining_fuel,
                new_rotation,
                new_thrust,
                dist_to_landing_spot,
                dist_to_surface,
                dist_to_path
        };
        return new_state;
    }

    std::vector<int> generate_random_action(int old_rota, int old_power_thrust) {
        std::vector<int> rotation_limits = this->_generate_action_limits(old_rota, 15, -90, 90);
        std::vector<int> thrust_limits = this->_generate_action_limits(old_power_thrust, 1, 0, 4);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis_rotation(rotation_limits[0], rotation_limits[1]);
        std::uniform_int_distribution<> dis_thrust(thrust_limits[0], thrust_limits[1]);
        int random_rotation = dis_rotation(gen);
        int random_thrust = dis_thrust(gen);
        return {random_rotation, random_thrust};
    }

    std::vector<int> _generate_action_limits(int center, int delta, int min_value, int max_value) {
        return {std::max(center - delta, min_value), std::min(center + delta, max_value)};
    }

    std::pair<float, float> calculate_intersection(std::pair<float, float> previous_pos,
                                                   std::pair<float, float> new_pos,
                                                   std::vector<std::pair<int, int>> surface) {
        float x1 = previous_pos.first;
        float y1 = previous_pos.second;
        float x2 = new_pos.first;
        float y2 = new_pos.second;
        std::vector<float> x3, y3, x4, y4;
        for (auto s : surface) {
            x3.push_back(s.first.first);
            y3.push_back(s.first.second);
            x4.push_back(s.second.first);
            y4.push_back(s.second.second);
        }
        std::vector<float> denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        std::vector<bool> mask = denominator != 0;
        std::vector<float> t(denominator.size());
        std::vector<float> u(denominator.size());
        for (int i = 0; i < denominator.size(); i++) {
            if (mask[i]) {
                t[i] = ((x1 - x3[i]) * (y3[i] - y4[i]) - (y1 - y3[i]) * (x3[i] - x4[i])) / denominator[i];
                u[i] = -((x1 - x2) * (y1 - y3[i]) - (y1 - y2) * (x1 - x3[i])) / denominator[i];
            }
        }
        std::vector<bool> intersected_mask = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1);
        std::pair<float, float> new_pos = new_pos;
        if (std::any_of(intersected_mask.begin(), intersected_mask.end(), [](bool i) { return i; })) {
            std::vector<float> intersection_x, intersection_y;
            for (int i = 0; i < intersected_mask.size(); i++) {
                if (intersected_mask[i]) {
                    intersection_x.push_back(x1 + t[i] * (x2 - x1));
                    intersection_y.push_back(y1 + t[i] * (y2 - y1));
                }
            }
            new_pos = {intersection_x[0], intersection_y[0]};
        }
        return new_pos;
    }

    float get_distance_to_path(std::pair<float, float> new_pos, std::vector<std::pair<int, int>> path_to_the_landing_spot) {
        std::pair<float, float> highest;
        for (auto point : path_to_the_landing_spot) {
            if (new_pos.second >= point.second && !this->distance_2(new_pos, point) < std::pow(25, 2)) {
                highest = point;
                this->i_intermediate_path = i;
                break;
            }
        }
        if (highest.empty()) {
            highest = path_to_the_landing_spot.back();
        }
        return this->distance_2(highest, new_pos);
    }

    int straighten_info(std::vector<float> state) {
        float x = state[0];
        float y = state[1];
        std::pair<float, float> highest;
        for (auto point : this->path_to_the_landing_spot) {
            if (y >= point.second) {
                highest = point;
                break;
            }
        }
        if (highest.empty()) {
            highest = this->path_to_the_landing_spot.back();
        }
        if (x < highest.first) {
            return -1;
        } else if (x > highest.first) {
            return 1;
        } else {
            return 0;
        }
    }

    float norm_reward(float feature, int interval_low, int interval_high) {
        feature = std::clamp(feature, interval_low, interval_high);
        return 1.0 - ((feature - interval_low) / (interval_high - interval_low));
    }

    bool do_segments_intersect(std::pair<std::pair<int, int>, std::pair<int, int>> segment1,
                               std::pair<std::pair<int, int>, std::pair<int, int>> segment2) {
        std::pair<int, int> p1 = segment1.first;
        std::pair<int, int> p2 = segment1.second;
        std::pair<int, int> p3 = segment2.first;
        std::pair<int, int> p4 = segment2.second;
        if (p1 == p3 || p1 == p4 || p2 == p3 || p2 == p4) {
            return false;
        }
        int o1 = this->orientation(p1, p2, p3);
        int o2 = this->orientation(p1, p2, p4);
        int o3 = this->orientation(p3, p4, p1);
        int o4 = this->orientation(p3, p4, p2);
        if ((o1 != o2 && o3 != o4) ||
            (o1 == 0 && this->on_segment(p1, p3, p2)) ||
            (o2 == 0 && this->on_segment(p1, p4, p2)) ||
            (o3 == 0 && this->on_segment(p3, p1, p4)) ||
            (o4 == 0 && this->on_segment(p3, p2, p4))) {
            if (!this->on_segment(p1, p2, p3) && !this->on_segment(p1, p2, p4)) {
                return true;
            }
        }
        return false;
    }

    int orientation(std::pair<int, int> p, std::pair<int, int> q, std::pair<int, int> r) {
        int val = (q.second - p.second) * (r.first - q.first) - (q.first - p.first) * (r.second - q.second);
        if (val == 0) {
            return 0;
        }
        return val > 0 ? 1 : 2;
    }

    bool on_segment(std::pair<int, int> p, std::pair<int, int> q, std::pair<int, int> r) {
        return (std::max(p.first, r.first) >= q.first >= std::min(p.first, r.first) &&
                std::max(p.second, r.second) >= q.second >= std::min(p.second, r.second));
    }
};

class GeneticAlgorithm {
private:
    RocketLandingEnv env;
    int horizon;
    int offspring_size;
    int n_elites;
    int n_heuristic_guides;
    float mutation_rate;
    int population_size;
    std::vector<std::vector<std::vector<int>>> population;
    std::vector<std::vector<std::vector<int>>> parents;

public:
    GeneticAlgorithm(RocketLandingEnv env) {
        this->env = env;
        this->horizon = 15;
        this->offspring_size = 9;
        this->n_elites = 3;
        this->n_heuristic_guides = 3;
        this->mutation_rate = 0.4;
        this->population_size = this->offspring_size + this->n_elites + this->n_heuristic_guides;
        this->population = this->init_population(this->env.initial_state[5], this->env.initial_state[6]);
        this->parents = {};
    }

    std::vector<std::vector<std::vector<int>>> crossover(std::vector<std::vector<std::vector<int>>> population_survivors,
                                                         int offspring_size) {
        std::vector<std::vector<std::vector<int>>> offspring;
        while (offspring.size() < offspring_size) {
            std::vector<int> indices = this->random_int(population_survivors.size(), 2);
            std::vector<std::vector<std::vector<int>>> parents = {population_survivors[indices[0]], population_survivors[indices[1]]};
            std::vector<std::vector<int>> policy(this->horizon, std::vector<int>(2, 0));
            for (int i = 0; i < this->horizon; i++) {
                int offspring_rotation = this->my_random_int(parents[0][i][0], parents[1][i][0]);
                int offspring_thrust = this->my_random_int(parents[0][i][1], parents[1][i][1]);
                offspring_rotation = std::clamp(offspring_rotation, -90, 90);
                offspring_thrust = std::clamp(offspring_thrust, 0, 4);
                policy[i] = {offspring_rotation, offspring_thrust};
            }
            offspring.push_back(policy);
        }
        return offspring;
    }

    std::vector<std::vector<std::vector<int>>> mutation(std::vector<std::vector<std::vector<int>>> population) {
        std::vector<std::vector<std::vector<int>>> mutated_population = population;
        std::vector<std::vector<int>> individual = mutated_population[this->random_int(mutated_population.size())[0]];
        for (auto& action : individual) {
            action[1] = 4;
        }
        for (auto& individual : mutated_population) {
            for (auto& action : individual) {
                if (this->random_float() < this->mutation_rate) {
                    action[0] += this->random_int(-15, 16)[0];
                    action[1] += this->random_int(-1, 2)[0];
                    action[0] = std::clamp(action[0], -90, 90);
                    action[1] = std::clamp(action[1], 0, 4);
                }
            }
        }
        return mutated_population;
    }

    std::vector<std::vector<std::vector<int>>> heuristic(std::vector<float> curr_initial_state) {
        std::vector<std::vector<std::vector<int>>> heuristic_guides(3, std::vector<std::vector<int>>(this->horizon, std::vector<int>(2, 0)));
        float x = curr_initial_state[0];
        float y = curr_initial_state[1];
        float angle = std::atan2(this->env.middle_landing_spot.second - y, this->env.middle_landing_spot.first - x);
        float angle_degrees = std::degrees(angle);
        for (auto& action : heuristic_guides[0]) {
            action[1] = 4;
            action[0] = std::clamp(std::round(-angle_degrees), -90.0f, 90.0f);
        }
        for (auto& action : heuristic_guides[1]) {
            action[1] = 4;
            action[0] = std::clamp(std::round(angle_degrees), -90.0f, 90.0f);
        }
        for (auto& action : heuristic_guides[2]) {
            action[1] = 4;
            action[0] = 0;
        }
        return heuristic_guides;
    }

    void learn(int time_available) {
        this->env.reset();
        std::vector<float> curr_initial_state = this->env.initial_state;
        this->population = this->init_population(curr_initial_state[5], curr_initial_state[6], this->parents);
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            std::vector<float> rewards;
            for (auto individual : this->population) {
                rewards.push_back(this->rollout(individual));
            }
            this->parents = this->selection(rewards);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            if (time_elapsed >= time_available) {
                break;
            }
            std::vector<std::vector<std::vector<int>>> heuristic_guides = this->heuristic(curr_initial_state);
            heuristic_guides.erase(std::remove_if(heuristic_guides.begin(), heuristic_guides.end(),
                                                  [&](std::vector<std::vector<int>> item) {
                                                      return std::any_of(this->parents.begin(), this->parents.end(),
                                                                         [&](std::vector<std::vector<int>> parent) {
                                                                             return std::equal(item.begin(), item.end(), parent.begin());
                                                                         });
                                                  }), heuristic_guides.end());
            int offspring_size = this->offspring_size + this->n_heuristic_guides - heuristic_guides.size();
            std::vector<std::vector<std::vector<int>>> offspring = this->crossover(this->parents, offspring_size);
            offspring = this->mutation(offspring);
            offspring = this->mutation_heuristic(offspring, curr_initial_state[7]);
            if (!heuristic_guides.empty()) {
                this->population.insert(this->population.end(), heuristic_guides.begin(), heuristic_guides.end());
            }
            this->population.insert(this->population.end(), offspring.begin(), offspring.end());
        }
        std::vector<std::vector<int>> best_individual = this->parents.back();
        this->env.reset();
        std::vector<int> action_to_do = this->final_heuristic_verification(best_individual[0], curr_initial_state);
        std::vector<float> next_state = this->env.step(action_to_do);
        this->env.initial_state = next_state;
        this->parents = this->parents.erase(this->parents.begin());
        std::vector<int> last_elements_tuple = this->parents.back().back();
        std::vector<std::vector<int>> new_individuals;
        for (auto item : last_elements_tuple) {
            new_individuals.push_back(this->env.generate_random_action(item[0], item[1]));
        }
        this->parents.push_back(new_individuals);
        std::cout << action_to_do[0] << " " << action_to_do[1] << std::endl;
    }

    float rollout(std::vector<std::vector<int>> policy) {
        this->env.reset();
        float reward = 0;
        for (auto action : policy) {
            std::vector<float> _, reward, terminated, truncated, _ = this->env.step(action);
            if (terminated || truncated) {
                break;
            }
        }
        return reward;
    }

    std::vector<std::vector<std::vector<int>>> init_population(int previous_rotation, int previous_thrust,
                                                               std::vector<std::vector<std::vector<int>>> parents = {}) {
        std::vector<std::vector<std::vector<int>>> population(this->population_size, std::vector<std::vector<int>>(this->horizon, std::vector<int>(2, 0)));
        int num_parents = parents.empty() ? 0 : parents.size();
        if (!parents.empty()) {
            for (int i = 0; i < num_parents; i++) {
                population[i] = parents[i];
            }
        }
        for (int i = num_parents; i < this->population_size; i++) {
            population[i] = this->generate_random_individual(previous_rotation, previous_thrust);
        }
        return population;
    }

    std::vector<std::vector<int>> generate_random_individual(int previous_rotation, int previous_thrust) {
        std::vector<std::vector<int>> individual(this->horizon, std::vector<int>(2, 0));
        std::vector<int> random_action = {previous_rotation, previous_thrust};
        for (int i = 0; i < this->horizon; i++) {
            random_action = this->env.generate_random_action(random_action[0], random_action[1]);
            individual[i] = random_action;
        }
        return individual;
    }

    std::vector<std::vector<std::vector<int>>> mutation_heuristic(std::vector<std::vector<std::vector<int>>> population,
                                                                  float dist_landing_spot) {
        if (dist_landing_spot < std::pow(300, 2)) {
            std::vector<std::vector<int>> individual = population[this->random_int(population.size())[0]];
            for (auto& action : individual) {
                action[0] = 0;
            }
        }
        return population;
    }

    std::vector<int> final_heuristic_verification(std::vector<int> action_to_do, std::vector<float> state) {
        int rotation = state[5];
        if (std::abs(rotation - action_to_do[0]) > 110) {
            action_to_do[1] = 0;
        }
        return action_to_do;
    }

    std::vector<std::vector<int>> selection(std::vector<float> rewards) {
        std::vector<int> sorted_indices(rewards.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&](int i, int j) { return rewards[i] < rewards[j]; });
        std::vector<std::vector<int>> parents;
        for (int i = sorted_indices.size() - this->n_elites; i < sorted_indices.size(); i++) {
            parents.push_back(this->population[sorted_indices[i]]);
        }
        return parents;
    }

    int my_random_int(int a, int b) {
        if (a == b) {
            return a;
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(std::min(a, b), std::max(a, b));
            return dis(gen);
        }
    }

    std::vector<int> random_int(int n, int k = 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        return std::vector<int>(indices.begin(), indices.begin() + k);
    }

    float random_float() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        return dis(gen);
    }
};

int main() {
    int n;
    std::cin >> n;
    std::vector<std::pair<int, int>> surface;
    for (int i = 0; i < n; i++) {
        int land_x, land_y;
        std::cin >> land_x >> land_y;
        surface.push_back({land_x, land_y});
    }
    RocketLandingEnv env(initial_state, surface);
    GeneticAlgorithm my_GA(env);
    int i2 = 0;
    while (true) {
        std::vector<int> state;
        for (int i = 0; i < 10; i++) {
            int s;
            std::cin >> s;
            state.push_back(s);
        }
        if (i2 == 0) {
            env = RocketLandingEnv(state, surface);
            my_GA = GeneticAlgorithm(env);
        }
        my_GA.learn(84);
        i2 += 1;
    }
    return 0;
}



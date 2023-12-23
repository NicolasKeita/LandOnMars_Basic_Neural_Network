#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <numeric>

int my_random_int(int a, int b) {
    if (a == b) {
        return a;
    } else {
        return rand() % (std::max(a, b) - std::min(a, b) + 1) + std::min(a, b);
    }
}

class GeneticAlgorithm;

class RocketLandingEnv {
public:
    std::vector<double> initialState;
    std::vector<int> middleLandingSpot;
    std::vector<double> state;
    std::vector<std::vector<std::vector<int>>> surfaceSegments;
    // Member variables...

//RocketLandingEnv(std::vector<int> initial_state, std::vector<std::vector<int>> surface);
    RocketLandingEnv(std::vector<double> initialState, std::vector<std::vector<int>> surface)
        : n_intermediate_path(6),
          initialState(std::vector<double>(10, 0.0)),  // Initialize to 0.0
          state(initialState),
          actionConstraints({15, 1}),
          gravity(3.711) {

        // Copy surface data
        this->surface = surface;

        // Populate surface_segments
        /*
        for (size_t i = 0; i < this->surface.size() - 1; ++i) {
            this->surfaceSegments.push_back({this->surface[i], this->surface[i + 1]});
        }*/
        for (size_t i = 0; i < this->surface.size() - 1; ++i) {
            this->surfaceSegments.push_back({this->surface[i], this->surface[i + 1]});
        }

        // Find landing spot
        this->landingSpot = findLandingSpot(this->surface);

        // Calculate middle landing spot
        this->middleLandingSpot = {
            static_cast<int>(std::round(std::accumulate(this->landingSpot.begin(), this->landingSpot.end(), 0.0) / this->landingSpot.size())),
            static_cast<int>(std::round(std::accumulate(this->landingSpot.begin() + 1, this->landingSpot.end(), 0.0) / this->landingSpot.size()))
        };

        // Search path to the landing spot
        this->pathToTheLandingSpot = searchPath(
            {initialState[0], initialState[1]},
            this->surface,
            this->landingSpot,
            {}
        );

        // Modify path to the landing spot
        for (size_t i = 0; i < this->pathToTheLandingSpot.size(); ++i) {

            int x = this->pathToTheLandingSpot[i][0];
            int y = (i < this->pathToTheLandingSpot.size() - 1) ? this->pathToTheLandingSpot[i][1] + 200 : this->pathToTheLandingSpot[i][1];
            this->pathToTheLandingSpot[i] = {x, y};
        }

        // Initialize initial_state
        this->initialState[0] = static_cast<double>(initialState[0]);  // x
        this->initialState[1] = static_cast<double>(initialState[1]);  // y
        this->initialState[2] = static_cast<double>(initialState[2]);  // horizontal speed
        this->initialState[3] = static_cast<double>(initialState[3]);  // vertical speed
        this->initialState[4] = static_cast<double>(initialState[4]);  // fuel remaining
        this->initialState[5] = static_cast<double>(initialState[5]);  // rotation
        this->initialState[6] = static_cast<double>(initialState[6]);  // thrust power
        this->initialState[7] = static_cast<double>(distance_to_line(initialState[0], initialState[1], {this->landingSpot}));  // distance to landing spot
        this->initialState[8] = static_cast<double>(distance_to_line(initialState[0], initialState[1], this->surfaceSegments));  // distance to surface
        this->initialState[9] = 0.0;  // Initialize to 0.0
    }

    float distance_to_line(float x, float y, const std::vector<std::vector<std::vector<int>>>& line_segments)
    {
        std::vector<float> x1, y1, x2, y2;
        for (const auto& segment : line_segments) {
            x1.push_back(segment[0][0]);
            y1.push_back(segment[0][1]);
            x2.push_back(segment[1][0]);
            y2.push_back(segment[1][1]);
        }

        std::vector<float> dx = x2;
        std::transform(x1.begin(), x1.end(), dx.begin(), dx.begin(), std::minus<float>());
        std::vector<float> dy = y2;
        std::transform(y1.begin(), y1.end(), dy.begin(), dy.begin(), std::minus<float>());

        std::vector<float> dot_product(line_segments.size());
        for (size_t i = 0; i < line_segments.size(); ++i) {
            dot_product[i] = (x - x1[i]) * dx[i] + (y - y1[i]) * dy[i];
        }

        std::vector<float> t(line_segments.size());
        for (size_t i = 0; i < line_segments.size(); ++i) {
            t[i] = std::clamp(dot_product[i] / (dx[i] * dx[i] + dy[i] * dy[i]), 0.0f, 1.0f);
        }

        std::vector<float> closest_point_x(line_segments.size());
        std::transform(x1.begin(), x1.end(), dx.begin(), closest_point_x.begin(),
                    [](float a, float b) { return a + b; });
        std::transform(t.begin(), t.end(), dx.begin(), closest_point_x.begin(),
                    [](float a, float b) { return a * b; });

        std::vector<float> closest_point_y(line_segments.size());
        std::transform(y1.begin(), y1.end(), dy.begin(), closest_point_y.begin(),
                    [](float a, float b) { return a + b; });
        std::transform(t.begin(), t.end(), dy.begin(), closest_point_y.begin(),
                    [](float a, float b) { return a * b; });

        std::vector<float> segment_distance_squared(line_segments.size());
        std::transform(x1.begin(), x1.end(), closest_point_x.begin(), segment_distance_squared.begin(),
                    [](float a, float b) { return (a - b) * (a - b); });
        std::transform(y1.begin(), y1.end(), closest_point_y.begin(), segment_distance_squared.begin(),
                    [](float a, float b) { return (a - b) * (a - b); });

        float min_distance_squared = *std::min_element(segment_distance_squared.begin(), segment_distance_squared.end());

        return min_distance_squared;
    }

    static std::vector<std::vector<int>> findLandingSpot(const std::vector<std::vector<int>>& planetSurface);

    void reset();

    std::tuple<std::vector<double>, double, bool, bool, bool> step(const std::array<int, 2>& action_to_do)
    {
        state = computeNextState(state, action_to_do);
        auto [reward, terminated, truncated] = computeReward(state);
        return {state, reward, terminated, truncated, false};
    }

    std::vector<double> computeNextState(const std::vector<double>& state, const std::array<int, 2>& action);

    std::tuple<float, bool, bool> computeReward(const std::vector<double>& state);

    std::array<int, 2> generateRandomAction(int oldRota, int oldPowerThrust) {
        auto rotationLimits = generateActionLimits(oldRota, 15, -90, 90);
        auto thrustLimits = generateActionLimits(oldPowerThrust, 1, 0, 4);

        int randomRotation = my_random_int(rotationLimits[0], rotationLimits[1]);
        int randomThrust = my_random_int(thrustLimits[0], thrustLimits[1]);

        return {randomRotation, randomThrust};
    }

    std::vector<std::vector<int>> searchPath(const std::vector<double>& initial_pos,
                                            const std::vector<std::vector<int>>& surface,
                                            const std::vector<std::vector<int>>& landing_spot,
                                            const std::vector<std::vector<int>>& my_path) {
        std::vector<std::vector<int>> path;

        for (const auto& s2 : surface) {
            std::vector<int> intersection;
            if (do_segments_intersect({initial_pos, landing_spot}, s2)) {
                intersection = (s2[0][1] > s2[1][1]) ? s2[0] : s2[1];
            }

            if (!intersection.empty()) {
                path.push_back(intersection);
                break;
            }
        }

        if (path.empty()) {
            const int n_intermediate_path = 6;
            std::vector<int> t;
            for (int i = 0; i < n_intermediate_path; ++i) {
                t = {(i * (initial_pos[0] + landing_spot[0]) / (n_intermediate_path - 1)),
                    (i * (initial_pos[1] + landing_spot[1]) / (n_intermediate_path - 1))};
                path.push_back(t);
            }

            if (!my_path.empty()) {
                path.insert(path.end(), my_path.begin(), my_path.end() - 1);
            }
        }

        return path;
    }

    int getDistanceToPath(const std::vector<int>& newPos,
                          const std::vector<std::vector<int>>& pathToTheLandingSpot);

    int straightenInfo(const std::vector<int>& state);

    // Member functions...

private:

    std::vector<std::vector<int>> surface;
    double gravity;
    int n_intermediate_path;
    std::vector<std::vector<int>> landingSpot;
    std::vector<std::vector<int>> pathToTheLandingSpot;

    std::vector<int> actionConstraints;

    double initialFuel;

    std::array<int, 2> generateActionLimits(int center, int delta, int minValue, int maxValue) {
        return {
            std::max(center - delta, minValue),
            std::min(center + delta, maxValue)
        };
    }

    std::tuple<int, int> limitActions(int oldRota, int oldPowerThrust, const std::vector<int>& action);

    double normReward(double feature, double intervalLow, double intervalHigh);
};

class GeneticAlgorithm {
public:
    RocketLandingEnv* env;
    int horizon;
    int offspring_size;
    int n_elites;
    int n_heuristic_guides;
    double mutation_rate;
    int population_size;
    std::vector<std::vector<std::array<int, 2>>> parents;

    GeneticAlgorithm(RocketLandingEnv* env)
        : env(env),
          horizon(15),
          offspring_size(9),
          n_elites(3),
          n_heuristic_guides(3),
          mutation_rate(0.4),
          population_size(offspring_size + n_elites + n_heuristic_guides),
          population(init_population(env->initialState[5], env->initialState[6])),
          parents(std::vector<std::vector<std::array<int, 2>>>()) {
        // Additional initialization if needed
    }

    void learn(int time_available) {
        env->reset();
        auto curr_initial_state = env->initialState;
        population = init_population(curr_initial_state[5], curr_initial_state[6], parents);
        auto start_time = std::time(0);

        while (true) {
            std::vector<double> rewards;
            for (const std::vector<std::array<int, 2>>& individual : population) {
                rewards.push_back(rollout(individual));
            }

            parents = selection(population, rewards, n_elites);

            if ((std::time(0) - start_time) * 1000 >= time_available) {
                break;
            }

            auto heuristic_guides = heuristic(curr_initial_state);

            heuristic_guides.erase(std::remove_if(heuristic_guides.begin(), heuristic_guides.end(),
                                                [this](const auto& item) {
                                                    return std::any_of(parents.begin(), parents.end(),
                                                                        [&item](const auto& parent) {
                                                                            return item == parent;
                                                                        });
                                                }), heuristic_guides.end());

            auto offspring_size = this->offspring_size + this->n_heuristic_guides - heuristic_guides.size();
            auto offspring = crossover(parents, offspring_size);
            offspring = mutation(offspring);
            offspring = mutation_heuristic(offspring, curr_initial_state[7]);

            if (!heuristic_guides.empty()) {
                population.insert(population.end(), heuristic_guides.begin(), heuristic_guides.end());
            }

            population.insert(population.end(), offspring.begin(), offspring.end());
        }

        auto best_individual = parents.back();
        env->reset();
        auto action_to_do = final_heuristic_verification(best_individual[0], curr_initial_state);
        //auto [next_state, _, _, _, _] = env->step(action_to_do);
        std::vector<double> next_state;
        double reward;
        bool terminated, truncated, unused;
        std::tie(next_state, reward, terminated, truncated, unused) = env->step(action_to_do);
        env->initialState = next_state;
        for (auto& individual : parents) {
            individual.erase(individual.begin()); // Remove the first element of each individual
        }
        //parents = std::vector(best_individual.begin() + 1, best_individual.end());
        auto last_elements_tuple = parents.back();
        //parents.insert(parents.end(), std::vector<std::vector<int>>({{env->generateRandomAction(last_elements_tuple[0], last_elements_tuple[1])}}));
        std::vector<std::array<int, 2>> randomActions;
        for (const auto& item : last_elements_tuple) {
            randomActions.push_back(env->generateRandomAction(item[0], item[1]));
        }
        std::vector<std::vector<std::array<int, 2>>> newElements(1, randomActions);
        auto newArray = std::vector<std::vector<std::array<int, 2>>>(newElements);
        for (auto& parent : parents) {
            parent.insert(parent.end(), newArray[0].begin(), newArray[0].end());
        }
        std::cout << action_to_do[0] << " " << action_to_do[1] << std::endl;
    }

    double rollout(const std::vector<std::array<int, 2>>& policy) {
        env->reset();
        double totalReward = 0.0;

        for (const std::array<int, 2>& action : policy) {
            std::tuple<std::vector<double>, bool, bool, bool, bool> stepResult = env->step(action);
            double reward;
            bool terminated, truncated, unused;
            std::tie(std::ignore, reward, terminated, truncated, std::ignore) = stepResult;
            //auto [_, reward, terminated, truncated, _] = env->step(action);
            //auto [std::ignore, reward, terminated, truncated, std::ignore] = env->step(action);

            totalReward += reward;

            if (terminated || truncated) {
                break;
            }
        }

        return totalReward;
    }

    std::vector<std::vector<std::array<int, 2>>> selection(
        const std::vector<std::vector<std::array<int, 2>>>& population,
        const std::vector<double>& rewards,
        int n_parents
    ) {
        // Initialize sortedIndices manually
        std::vector<size_t> sortedIndices(rewards.size());
        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            sortedIndices[i] = i;
        }

        // Sort the indices based on rewards
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                [&rewards](size_t i, size_t j) { return rewards[i] < rewards[j]; });

        // Extract the top n_parents individuals from the sorted population
        std::vector<std::vector<std::array<int, 2>>> parents(n_parents);
        for (int i = 0; i < n_parents; ++i) {
            parents[i] = population[sortedIndices[sortedIndices.size() - n_parents + i]];
        }

        return parents;
    }

    std::vector<std::vector<std::array<int, 2>>> heuristic(const std::vector<double>& curr_initial_state)
    {
        std::vector<std::vector<std::array<int, 2>>> heuristics_guides(3, std::vector<std::array<int, 2>>(horizon, {0, 0}));

        int x = curr_initial_state[0];
        int y = curr_initial_state[1];

        double angle = std::atan2(env->middleLandingSpot[1] - y, env->middleLandingSpot[0] - x);
        double angle_degrees = std::round(angle * (180.0 / M_PI));

        for (auto& action : heuristics_guides[0]) {
            action[1] = 4;
            action[0] = std::clamp(static_cast<int>(-angle_degrees), -90, 90);
        }

        for (auto& action : heuristics_guides[1]) {
            action[1] = 4;
            action[0] = std::clamp(static_cast<int>(angle_degrees), -90, 90);
        }

        for (auto& action : heuristics_guides[2]) {
            action[1] = 4;
            action[0] = 0;
        }

        return heuristics_guides;
    }

    std::vector<std::vector<std::array<int, 2>>> crossover(const std::vector<std::vector<std::array<int, 2>>>& population_survivors, int offspring_size)
    {
        std::vector<std::vector<std::array<int, 2>>> offspring;

        while (offspring.size() < offspring_size) {
            std::vector<std::array<int, 2>> policy(horizon, {0, 0});
            std::vector<int> indices(2);

            for (int i = 0; i < 2; ++i) {
                indices[i] = rand() % population_survivors.size();
            }

            for (int i = 0; i < horizon; ++i) {
                int offspring_rotation = my_random_int(population_survivors[indices[0]][i][0], population_survivors[indices[1]][i][0]);
                int offspring_thrust = my_random_int(population_survivors[indices[0]][i][1], population_survivors[indices[1]][i][1]);
                offspring_rotation = std::clamp(offspring_rotation, -90, 90);
                offspring_thrust = std::clamp(offspring_thrust, 0, 4);
                policy[i] = {offspring_rotation, offspring_thrust};
            }

            offspring.push_back(policy);
        }

        return offspring;
    }

    std::vector<std::vector<std::array<int, 2>>> mutation(const std::vector<std::vector<std::array<int, 2>>>& population)
    {
        std::vector<std::vector<std::array<int, 2>>> mutatedPopulation = population;

        // Select a random individual from the population
        std::vector<std::array<int, 2>>& individual = mutatedPopulation[std::rand() % mutatedPopulation.size()];

        // Set the second element of each action in the selected individual to 4
        for (auto& action : individual) {
            action[1] = 4;
        }

        // Mutate the entire population
        for (auto& ind : mutatedPopulation) {
            for (auto& action : ind) {
                // Apply mutation with a certain probability
                if (static_cast<double>(std::rand()) / RAND_MAX < mutation_rate) {
                    action[0] += std::rand() % 31 - 15;  // Random value in the range [-15, 15]
                    action[1] += std::rand() % 3 - 1;    // Random value in the range [-1, 1]

                    // Clip values to specified ranges
                    action[0] = std::clamp(action[0], -90, 90);
                    action[1] = std::clamp(action[1], 0, 4);
                }
            }
        }
        return mutatedPopulation;
    }

    std::vector<std::vector<std::array<int, 2>>> mutation_heuristic(const std::vector<std::vector<std::array<int, 2>>>& population,
                                                                  double dist_landing_spot)
    {
        if (dist_landing_spot < 300 * 300) {
            std::vector<std::vector<std::array<int, 2>>> mutatedPopulation = population;

            // Select a random individual from the population
            std::vector<std::array<int, 2>>& individual = mutatedPopulation[std::rand() % mutatedPopulation.size()];

            // Set the first element of each action in the selected individual to 0
            for (auto& action : individual) {
                action[0] = 0;
            }

            return mutatedPopulation;
        }

        return population;
    }

    std::array<int, 2> final_heuristic_verification(const std::array<int, 2>& action_to_do, const std::vector<double>& state) {
        int rotation = state[5];
        std::array<int, 2> result = action_to_do;

        if (std::abs(rotation - result[0]) > 110) {
            result[1] = 0;
        }

        return result;
    }

    // Other member functions...

private:
    std::vector<std::vector<std::array<int, 2>>> population;

    std::vector<std::vector<std::array<int, 2>>> init_population(int previous_rotation, int previous_thrust,
                                                                const std::vector<std::vector<std::array<int, 2>>>& parents = {}) {
        std::vector<std::vector<std::array<int, 2>>> population(population_size, std::vector<std::array<int, 2>>(horizon, {0, 0}));
        int num_parents = parents.size();

        if (num_parents > 0) {
            for (int i = 0; i < num_parents; ++i) {
                population[i] = parents[i];
            }
        }

        for (int i = num_parents; i < population_size; ++i) {
            population[i] = generate_random_individual(previous_rotation, previous_thrust);
        }

        return population;
    }

    std::vector<std::array<int, 2>> generate_random_individual(int previous_rotation, int previous_thrust) {
        std::vector<std::array<int, 2>> individual(horizon, {0, 0});
        std::array<int, 2> random_action = {previous_rotation, previous_thrust};
        for (int i = 0; i < horizon; ++i) {
            random_action = env->generateRandomAction(random_action[0], random_action[1]);
            individual[i] = random_action;
        }
        return individual;
    }

    // Other private member functions...
};

int main() {
    int n;
    std::cin >> n;

    std::vector<std::vector<int>> surface;
    for (int i = 0; i < n; ++i) {
        int land_x, land_y;
        std::cin >> land_x >> land_y;
        surface.push_back({land_x, land_y});
    }

    RocketLandingEnv* env = nullptr;
    GeneticAlgorithm* my_GA = nullptr;
    int i2 = 0;

    while (true) {
        std::vector<int> state;
        for (int i = 0; i < 8; ++i) {
            int value;
            std::cin >> value;
            state.push_back(value);
        }

        if (i2 == 0) {
            env = new RocketLandingEnv(state, surface);
            my_GA = new GeneticAlgorithm(env);
        }

        my_GA->learn(84);
        i2 += 1;
    }

    return 0;
}
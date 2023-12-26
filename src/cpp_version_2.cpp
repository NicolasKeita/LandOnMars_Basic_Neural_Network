#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <numeric>
#include <chrono>

int orientation(const std::tuple<int, int>& p, const std::tuple<int, int>& q, const std::tuple<int, int>& r) {
    int val = (std::get<1>(q) - std::get<1>(p)) * (std::get<0>(r) - std::get<0>(q)) - (std::get<0>(q) - std::get<0>(p)) * (std::get<1>(r) - std::get<1>(q));
    if (val == 0) {
        return 0;
    }
    return (val > 0) ? 1 : 2;
}

bool on_segment(const std::tuple<int, int>& p, const std::tuple<int, int>& q, const std::tuple<int, int>& r) {
    return (std::max(std::get<0>(p), std::get<0>(r)) >= std::get<0>(q) && std::get<0>(q) >= std::min(std::get<0>(p), std::get<0>(r)) &&
            std::max(std::get<1>(p), std::get<1>(r)) >= std::get<1>(q) && std::get<1>(q) >= std::min(std::get<1>(p), std::get<1>(r)));
}

bool do_segments_intersect(const std::vector<std::vector<double>>& segment1,
                           const std::vector<std::vector<double>>& segment2)
{
    double x1 = segment1[0][0];
    double y1 = segment1[0][1];
    double x2 = segment1[1][0];
    double y2 = segment1[1][1];

    double x3 = segment2[0][0];
    double y3 = segment2[0][1];
    double x4 = segment2[1][0];
    double y4 = segment2[1][1];

    if (std::tie(x1, y1) == std::tie(x3, y3) || std::tie(x1, y1) == std::tie(x4, y4) ||
        std::tie(x2, y2) == std::tie(x3, y3) || std::tie(x2, y2) == std::tie(x4, y4)) {
        return false;
    }

    int o1 = orientation({x1, y1}, {x2, y2}, {x3, y3});
    int o2 = orientation({x1, y1}, {x2, y2}, {x4, y4});
    int o3 = orientation({x3, y3}, {x4, y4}, {x1, y1});
    int o4 = orientation({x3, y3}, {x4, y4}, {x2, y2});

    if ((o1 != o2 && o3 != o4) || (o1 == 0 && on_segment({x1, y1}, {x3, y3}, {x2, y2})) ||
        (o2 == 0 && on_segment({x1, y1}, {x4, y4}, {x2, y2})) ||
        (o3 == 0 && on_segment({x3, y3}, {x1, y1}, {x4, y4})) ||
        (o4 == 0 && on_segment({x3, y3}, {x2, y2}, {x4, y4}))) {
        if (!on_segment({x1, y1}, {x2, y2}, {x3, y3}) && !on_segment({x1, y1}, {x2, y2}, {x4, y4})) {
            return true;
        }
    }
    return false;
}


int my_random_int(int a, int b) {
    if (a == b) {
        return a;
    } else {
        return rand() % (std::max(a, b) - std::min(a, b) + 1) + std::min(a, b);
    }
}

std::vector<std::vector<double>> interpolatePoints(const std::vector<std::vector<double>>& points, int numIntermediatePoints) {
    std::vector<std::vector<double>> result;

    if (points.size() < 2) {
        // Not enough points for interpolation
        return points;
    }

    for (size_t i = 0; i < points.size() - 1; ++i) {
        double startX = points[i][0];
        double startY = points[i][1];
        double endX = points[i + 1][0];
        double endY = points[i + 1][1];

        for (int j = 0; j <= numIntermediatePoints; ++j) {
            double x = startX + j * (endX - startX) / numIntermediatePoints;
            double y = startY + j * (endY - startY) / numIntermediatePoints;
            result.push_back({x, y});
        }
    }

    // Add the last point
    result.push_back(points.back());

    return result;
}

class GeneticAlgorithm;

class RocketLandingEnv {
public:
    std::vector<double> initialState;
    std::vector<double> state;
    std::vector<double> middleLandingSpot;
    std::vector<std::vector<std::vector<double>>> surfaceSegments;

    RocketLandingEnv(std::vector<double> initialStateInput, std::vector<std::vector<double>> surface)
        : n_intermediate_path(6),
          initialState(std::vector<double>(10, 0.0)),
          state(initialStateInput),
          middleLandingSpot(2, 0.0),
          actionConstraints({15, 1}),
          gravity(3.711)
    {

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

        this->middleLandingSpot[0] = landingSpot[0][0];
        this->middleLandingSpot[1] = (landingSpot[0][1] + landingSpot[1][1]) / 2.0;

        this->pathToTheLandingSpot = searchPath({initialStateInput[0], initialStateInput[1]});
        this->pathToTheLandingSpot.insert(this->pathToTheLandingSpot.begin(), {initialStateInput[0], initialStateInput[1]});
        this->pathToTheLandingSpot = interpolatePoints(this->pathToTheLandingSpot, this->n_intermediate_path);
        for (const auto& point : this->pathToTheLandingSpot) {
            //std::cerr << "X: " << point[0] << ", Y: " << point[1] << std::endl;
        }

        // Modify path to the landing spot
        for (size_t i = 0; i < this->pathToTheLandingSpot.size(); ++i) {

            double x = this->pathToTheLandingSpot[i][0];
            double y = (i < this->pathToTheLandingSpot.size() - 1) ? this->pathToTheLandingSpot[i][1] + 200 : this->pathToTheLandingSpot[i][1];
            this->pathToTheLandingSpot[i] = {x, y};
        }
        for (const auto& point : this->pathToTheLandingSpot) {
            std::cerr << "X: " << point[0] << ", Y: " << point[1] << std::endl;
        }


        // Initialize initial_state
        this->initialState[0] = static_cast<double>(initialStateInput[0]);  // x
        this->initialState[1] = static_cast<double>(initialStateInput[1]);  // y
        this->initialState[2] = static_cast<double>(initialStateInput[2]);  // horizontal speed
        this->initialState[3] = static_cast<double>(initialStateInput[3]);  // vertical speed
        this->initialState[4] = static_cast<double>(initialStateInput[4]);  // fuel remaining
        this->initialState[5] = static_cast<double>(initialStateInput[5]);  // rotation
        this->initialState[6] = static_cast<double>(initialStateInput[6]);  // thrust power
        this->initialState[7] = static_cast<double>(distance_to_line(initialStateInput[0], initialStateInput[1], {this->landingSpot}));  // distance to landing spot
        this->initialState[8] = static_cast<double>(distance_to_line(initialStateInput[0], initialStateInput[1], this->surfaceSegments));  // distance to surface
        this->initialState[9] = 0.0;
    }

    double distance_to_line(double x, double y, const std::vector<std::vector<std::vector<double>>>& lineSegments) {
        double minDistanceSquared = std::numeric_limits<double>::infinity();

        for (const auto& segment : lineSegments) {
            for (size_t i = 0; i < segment.size() - 1; ++i) {
                const std::vector<double>& point1 = segment[i];
                const std::vector<double>& point2 = segment[i + 1];

                double dx = point2[0] - point1[0];
                double dy = point2[1] - point1[1];

                double dotProduct = (x - point1[0]) * dx + (y - point1[1]) * dy;
                double t = std::max(0.0, std::min(1.0, dotProduct / (dx * dx + dy * dy)));
                std::vector<double> closestPoint = {point1[0] + t * dx, point1[1] + t * dy};

                double segmentDistanceSquared = distance_2({x, y}, closestPoint);
                minDistanceSquared = std::min(minDistanceSquared, segmentDistanceSquared);
            }
        }

        return minDistanceSquared;
    }

    static std::vector<std::vector<double>> findLandingSpot(const std::vector<std::vector<double>>& planetSurface)
    {
        for (size_t i = 0; i < planetSurface.size() - 1; ++i) {
            const auto& point1 = planetSurface[i];
            const auto& point2 = planetSurface[i + 1];

            if (point1[1] == point2[1]) {
                return {point1, point2};
            }
        }
        throw std::runtime_error("No landing site on test-case data");
    }

    std::pair<std::vector<double>, bool> reset() {
        state = initialState;
        return {state, false};
    }

    std::tuple<std::vector<double>, double, bool, bool, bool> step(const std::array<int, 2>& action_to_do)
    {
        state = computeNextState(state, action_to_do);
        auto [reward, terminated, truncated] = computeReward(state);
        return {state, reward, terminated, truncated, false};
    }

    double distance_2(const std::vector<double>& point1, const std::vector<double>& point2) {
        double dx = point1[0] - point2[0];
        double dy = point1[1] - point2[1];
        return dx * dx + dy * dy;
    }

    double getDistanceToPath(const std::vector<double>& newPos, const std::vector<std::vector<double>>& pathToTheLandingSpot) {
        //const double threshold = 25.0 * 25.0;

        std::vector<double> highest = {0.0, 0.0};

        for (std::size_t i = 0; i < pathToTheLandingSpot.size(); ++i) {
            const auto& point = pathToTheLandingSpot[i];
            if (newPos[1] >= point[1] /*&& distance_2(newPos, point) >= threshold*/) {
                highest = point;
                break;
            }
        }

        if (highest[0] == 0.0 && highest[1] == 0.0) {
            highest = pathToTheLandingSpot.back();
        }
        //std::cerr << "here height" << highest[0] << ' ' << highest[1] << ' ' <<newPos[0] << ' ' <<newPos[1] << std::endl;
        return distance_2(highest, newPos);
    }

    std::vector<double> computeNextState(const std::vector<double>& state, const std::array<int, 2>& action) {
        double x = state[0];
        double y = state[1];
        double hs = state[2];
        double vs = state[3];
        double remainingFuel = state[4];
        double rotation = state[5];
        double thrust = state[6];

        std::tie(rotation, thrust) = limitActions(rotation, thrust, action);

        double radians = M_PI / 180.0 * rotation;
        double xAcceleration = std::sin(radians) * thrust;
        double yAcceleration = std::cos(radians) * thrust - gravity;

        double newHorizontalSpeed = hs - xAcceleration;
        double newVerticalSpeed = vs + yAcceleration;

        double newX = x + hs - 0.5 * xAcceleration;
        double newY = y + vs + 0.5 * yAcceleration;

        std::vector<double> newPos = {std::clamp(newX, 0.0, 7000.0), std::clamp(newY, 0.0, 3000.0)};
        newPos = calculateIntersection({x, y}, newPos, surface);

        remainingFuel = std::max(remainingFuel - thrust, 0.0);


        double distToLandingSpot = distance_to_line(newPos[0], newPos[1], {landingSpot});
        double distToSurface = distance_to_line(newPos[0], newPos[1], surfaceSegments);
        double distToPath = getDistanceToPath(newPos, pathToTheLandingSpot);

        //std::cerr << "distToLandingSpot " << distToLandingSpot << std::endl;
        //std::cerr << "distToPath " << distToPath << std::endl;

        return {newPos[0], newPos[1], newHorizontalSpeed, newVerticalSpeed,
                remainingFuel, rotation, thrust, distToLandingSpot,
                distToSurface, distToPath};
    }

    std::vector<double> calculateIntersection(const std::vector<double>& previousPos,
                                            const std::vector<double>& newPos,
                                            const std::vector<std::vector<double>>& surface) {
        double x1 = previousPos[0];
        double y1 = previousPos[1];
        double x2 = newPos[0];
        double y2 = newPos[1];

        std::vector<double> x3(surface.size() - 1), y3(surface.size() - 1);
        std::vector<double> x4(surface.size() - 1), y4(surface.size() - 1);

        for (size_t i = 0; i < surface.size() - 1; ++i) {
            x3[i] = surface[i][0];
            y3[i] = surface[i][1];
            x4[i] = surface[i + 1][0];
            y4[i] = surface[i + 1][1];
        }

        std::vector<double> denominator(surface.size() - 1);
        std::vector<bool> mask(surface.size() - 1);

        for (size_t i = 0; i < surface.size() - 1; ++i) {
            denominator[i] = (x1 - x2) * (y3[i] - y4[i]) - (y1 - y2) * (x3[i] - x4[i]);
            mask[i] = denominator[i] != 0;
        }

        std::vector<double> t(surface.size() - 1);
        std::vector<double> u(surface.size() - 1);

        for (size_t i = 0; i < surface.size() - 1; ++i) {
            if (mask[i]) {
                t[i] = ((x1 - x3[i]) * (y3[i] - y4[i]) - (y1 - y3[i]) * (x3[i] - x4[i])) / denominator[i];
                u[i] = -((x1 - x2) * (y1 - y3[i]) - (y1 - y2) * (x1 - x3[i])) / denominator[i];
            }
        }

        std::vector<bool> intersectedMask(surface.size() - 1);

        for (size_t i = 0; i < surface.size() - 1; ++i) {
            intersectedMask[i] = (0 <= t[i]) && (t[i] <= 1) && (0 <= u[i]) && (u[i] <= 1);
        }

        if (std::any_of(intersectedMask.begin(), intersectedMask.end(), [](bool val) { return val; })) {
            size_t index = std::find(intersectedMask.begin(), intersectedMask.end(), true) - intersectedMask.begin();
            double intersectionX = x1 + t[index] * (x2 - x1);
            double intersectionY = y1 + t[index] * (y2 - y1);
            return {intersectionX, intersectionY};
        }

        return newPos;
    }

    //std::tuple<float, bool, bool> computeReward(const std::vector<double>& state);
    double norm_reward(double value, double target, double scale) {
        return std::exp(-0.5 * std::pow((value - target) / scale, 2));
    }

    std::tuple<double, bool, bool> computeReward(const std::vector<double>& state) {
        double x = state[0];
        double y = state[1];
        double hs = state[2];
        double vs = state[3];
        double remainingFuel = state[4];
        double rotation = state[5];
        double thrust = state[6];
        double distLandingSpot = state[7];
        double distSurface = state[8];
        double distPath = state[9];

        bool isSuccessfulLanding = distLandingSpot < 1 && rotation == 0 && std::abs(vs) <= 40 && std::abs(hs) <= 20;
        bool isCrashedOnLandingSpot = distLandingSpot < 1;
        bool isCrashedAnywhere = y <= 1 || y >= 3000 - 1 || x <= 1 || x >= 7000 - 1 || distSurface < 1 || remainingFuel < 4;
        bool isCloseToLand = distLandingSpot < 1500 * 1500;

        distPath = isCloseToLand ? distLandingSpot : distPath;

        /*
        std::cerr << "distance : " << (norm_reward(distPath, 0, 7500 * 7500)) << std::endl;
        std::cerr << "distance_2 : " << distPath << std::endl;
        std::cerr << "distance_4 : " << distLandingSpot << std::endl;
        */

        double reward = (norm_reward(distPath, 0, 7500 * 7500)
                        + 0.65 * norm_reward(std::abs(vs), 39, 140)
                        + 0.35 * norm_reward(std::abs(hs), 19, 140));

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



    std::array<int, 2> generateRandomAction(int oldRota, int oldPowerThrust) {
        auto rotationLimits = generateActionLimits(oldRota, 15, -90, 90);
        auto thrustLimits = generateActionLimits(oldPowerThrust, 1, 0, 4);

        int randomRotation = my_random_int(rotationLimits[0], rotationLimits[1]);
        int randomThrust = my_random_int(thrustLimits[0], thrustLimits[1]);

        return {randomRotation, randomThrust};
    }

    std::vector<std::vector<double>> searchPath(const std::vector<double>& initial_pos)
    {
        for (const std::vector<std::vector<double>>& segment : surfaceSegments) {

            if (do_segments_intersect({initial_pos, middleLandingSpot}, segment)) {
                auto intersection = (segment[0][1] > segment[1][1]) ? segment[0] : segment[1];
                auto result = searchPath(intersection);
                result.insert(result.begin(), initial_pos);
                return result;
            }
        }
        return std::vector<std::vector<double>>(1, middleLandingSpot);
        /*
            if (!intersection.empty()) {
                path = intersection;
                break;
            }
        }

        if (path.empty()) {
            std::vector<double> t;
            for (int i = 0; i < n_intermediate_path; ++i) {
                double x = i * (initial_pos[0] + landing_spot[0]) / (n_intermediate_path - 1);
                double y = i * (initial_pos[1] + landing_spot[1]) / (n_intermediate_path - 1);
                t = {x, y};
                path.push_back(t);
            }

            if (!my_path.empty()) {
                path.insert(path.end(), my_path.end() - 1, my_path.end());
            }
        }

        return path;
        */
    }

    int straightenInfo(const std::vector<int>& state);

    // Member functions...

private:

    std::vector<std::vector<double>> surface;
    double gravity;
    int n_intermediate_path;
    std::vector<std::vector<double>> landingSpot;
    std::vector<std::vector<double>> pathToTheLandingSpot;

    std::vector<int> actionConstraints;

    double initialFuel;

    std::array<int, 2> generateActionLimits(int center, int delta, int minValue, int maxValue) {
        return {
            std::max(center - delta, minValue),
            std::min(center + delta, maxValue)
        };
    }

    std::tuple<int, int> limitActions(int oldRota, int oldPowerThrust, const std::array<int, 2>& action) {
        auto rotationLimits = generateActionLimits(oldRota, 15, -90, 90);
        auto thrustLimits = generateActionLimits(oldPowerThrust, 1, 0, 4);

        int rotation = std::clamp(action[0], std::get<0>(rotationLimits), std::get<1>(rotationLimits));
        int thrust = std::clamp(action[1], std::get<0>(thrustLimits), std::get<1>(thrustLimits));

        return {rotation, thrust};
    }

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
          horizon(30),
          offspring_size(10),
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
        auto start_time = std::chrono::steady_clock::now();

        while (true) {
            parents.clear();
            /*
            std::cerr << "Parents after selection:" << std::endl;
            for (const auto& parent : parents) {
                std::cerr << "[";
                for (std::size_t j = 0; j < std::min(parent.size(), static_cast<std::size_t>(4)); ++j) {
                    const auto& gene = parent[j];
                    std::cerr << "[" << gene[0] << ", " << gene[1] << "] ";
                }
                std::cerr << "...]" << std::endl;
            }
            std::cerr << "Parent size : " << parents.size() << std::endl;
            */

            std::vector<double> rewards;
            auto start_time_2 = std::chrono::steady_clock::now();
            for (const std::vector<std::array<int, 2>>& individual : population) {
                rewards.push_back(rollout(individual));
            }


            for (std::size_t i = 0; i < population.size(); ++i) {
                const std::vector<std::array<int, 2>>& individual = population[i];
                double reward = rewards[i];
                /*
                std::cerr << "Individual " << i << ": ";
                for (std::size_t j = 0; j < std::min(individual.size(), static_cast<std::size_t>(4)); ++j) {
                    const auto& gene = individual[j];
                    std::cerr << "[" << gene[0] << ", " << gene[1] << "] ";
                }
                std::cerr << "... ";
                */

                /*
                std::cerr << "Individual " << i << ": ";
                for (std::size_t j = 0; j < individual.size(); ++j) {
                    const auto& gene = individual[j];
                    std::cerr << "[" << gene[0] << ", " << gene[1] << "] ";
                }
                std::cerr << "... ";
                    std::cerr << "Reward: " << reward << std::endl;
                  */
            }
            //std::cerr << "population size : " << population.size() << std::endl;

            auto end_time2 = std::chrono::steady_clock::now();
            auto elapsed_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time_2).count();
            //std::cerr << "Time taken for the loop: " << elapsed_time2 << " milliseconds" << std::endl;

            parents = selection(population, rewards, n_elites);

            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count();
            if (elapsed_time >= time_available) {
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

            //std::cerr << offspring_size << std::endl;
            auto offspring_size = this->offspring_size + this->n_heuristic_guides - heuristic_guides.size();
            //std::cerr << offspring_size << std::endl;
            auto offspring = crossover(parents, offspring_size);
            //std::cerr << offspring.size() << " " <<  std::endl;
            offspring = mutation(offspring);
            offspring = mutation_heuristic(offspring, curr_initial_state[7]);

            population.clear();
            population.insert(population.end(), heuristic_guides.begin(), heuristic_guides.end());
            population.insert(population.end(), offspring.begin(), offspring.end());
            population.insert(population.end(), parents.begin(), parents.end());
        }
        auto best_individual = parents.back();
        env->reset();
        auto action_to_do = final_heuristic_verification(best_individual[0], curr_initial_state);
        std::vector<double> next_state;
        double reward;
        bool terminated, truncated, unused;
        std::tie(next_state, reward, terminated, truncated, unused) = env->step(action_to_do);
        env->initialState = next_state;
        for (auto& individual : parents) {
            individual.erase(individual.begin());
        }

        std::vector<std::array<int, 2>> last_elements_tuple;
        for (const std::vector<std::array<int, 2>>& parent : parents) {
            last_elements_tuple.push_back(parent.back());
        }
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

        for (const auto& action : policy) {
            auto [_, reward, terminated, truncated, __] = env->step(action);
            totalReward = reward;
            if (terminated || truncated) break;
        }
        //std::cerr << totalReward << std::endl;
        return totalReward;
    }


    std::vector<std::vector<std::array<int, 2>>> selection(
        const std::vector<std::vector<std::array<int, 2>>>& population,
        const std::vector<double>& rewards,
        int n_parents
    )
    {
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

    std::vector<std::vector<double>> surface;
    for (int i = 0; i < n; ++i) {
        double land_x, land_y;
        std::cin >> land_x >> land_y;
        surface.push_back({land_x, land_y});
    }

    RocketLandingEnv* env = nullptr;
    GeneticAlgorithm* my_GA = nullptr;
    int i2 = 0;

    while (true) {
        std::vector<double> state;
        double x;
        double y;
        double hs;
        double vs;
        double f;
        double r;
        double p;
        std::cin >> x >> y >> hs >> vs >> f >> r >> p; std::cin.ignore();
        state = {x, y, hs, vs, f, r, p};
        if (i2 == 0) {
            env = new RocketLandingEnv(state, surface);
            my_GA = new GeneticAlgorithm(env);
        }

        my_GA->learn(95);
        i2 += 1;
    }

    return 0;
}
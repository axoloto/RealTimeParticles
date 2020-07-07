
#ifndef BOIDS_GENERATOR_HPP
#define BOIDS_GENERATOR_HPP

namespace Core {
    class BoidsGenerator {
        public:
            BoidsGenerator(size_t id);
            ~BoidsGenerator() = default;
        private:
            size_t m_id;
    };
}

#endif


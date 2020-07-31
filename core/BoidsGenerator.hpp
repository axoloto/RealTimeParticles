
#pragma once 

#include<array>

namespace Core {

    static constexpr int NUM_ENTITIES = 3000;

    class BoidsGenerator {
        public:
            BoidsGenerator(int halfBoxSize);
            ~BoidsGenerator() = default;

            void updateBoids();

            void* getVerticesBufferStart();
            size_t getVerticesBufferSize();

        private:

            void generateBoids();

            // WIP
            struct Entity
            {
                std::array<float, 3> xyz;
                std::array<float, 3> rgb;
            };
            std::array<Entity, NUM_ENTITIES> m_entities;

            int m_halfBoxSize;
    };
}


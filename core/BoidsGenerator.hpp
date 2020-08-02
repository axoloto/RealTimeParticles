
#pragma once 

#include<array>

namespace Core {

    static constexpr int NUM_MAX_ENTITIES = 3000;

    class BoidsGenerator {
        public:
            BoidsGenerator(int boxSize, int numEntities);
            ~BoidsGenerator() = default;

            void updateBoids();

            void* getVerticesBufferStart();
            size_t getVerticesBufferSize();

            int numEntities() const { return m_numEntities; }
            void setNumEntities(int numEntities) { m_numEntities = numEntities; }

        private:

            void generateBoids();

            // WIP
            struct Entity
            {
                std::array<float, 3> xyz;
                std::array<float, 3> rgb;
            };
            std::array<Entity, NUM_MAX_ENTITIES> m_entities;

            int m_numEntities;
            int m_boxSize;
    };
}


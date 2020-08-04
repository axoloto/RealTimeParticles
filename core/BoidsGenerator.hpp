
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
            float randomZeroOne();
            float norm3D(float x, float y,float z);
            void UpdateCoord();
            void WallCyclic();
            // WIP
            struct Entity
            {
                std::array<float, 3> xyz;
                std::array<float, 3> rgb;                
            };
            struct Phy_boid
            {
                std::array<float, 3> xyz;
                std::array<float, 3> vxyz;
                //std::array<float, 3> vxyz;
            };
            std::array<Entity, NUM_MAX_ENTITIES> m_entities;
            std::array<Phy_boid, NUM_MAX_ENTITIES> m_phy_boids;

            int m_numEntities;
            int m_boxSize;
    };
}


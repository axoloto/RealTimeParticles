#pragma once 

#include<array>
#include"diligentGraphics/Math.hpp"

namespace Core {

    static constexpr int NUM_MAX_ENTITIES = 30000;

    class Physics {
        public:
            Physics(int boxSize, int numEntities);
            ~Physics() = default;

            void* getCoordsBufferStart();
            void* getColorsBufferStart();

            int numEntities() const { return m_numEntities; }
            void setNumEntities(int numEntities) { m_numEntities = numEntities; }

            virtual void updatePhysics() = 0;

            void update();
            void updateBuffers();
            
        protected:

            struct Entity
            {
                // Mandatory
                Math::float3 xyz;
                // Mandatory
                Math::float3 rgb;

                Math::float3 vxyz;
                Math::float3 axyz;
            };

            std::array<Entity, NUM_MAX_ENTITIES> m_entities;
            
            std::array<std::array<float, 3>, NUM_MAX_ENTITIES> m_coordsBuffer;
            std::array<std::array<float, 3>, NUM_MAX_ENTITIES> m_colorsBuffer;

            int m_numEntities;
            int m_boxSize;
            
    };
}

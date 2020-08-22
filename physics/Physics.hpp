#pragma once 

#include<array>
#include"diligentGraphics/Math.hpp"

namespace Core {

    static constexpr int NUM_MAX_ENTITIES = 30000;

    enum Dimension {dim2D, dim3D}; 

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

            void setForcedMaxSpeed(bool forcedmax) { m_forcedmaxspeed = forcedmax; }
            float getForcedMaxspeed() { return m_forcedmaxspeed; }
            void setPause(bool pause) { m_pause = pause; }
            float getPause() { return m_pause; }
            void setBouncingWall(bool bouncingwall) { m_activateBouncingWall = bouncingwall; }
            float getBouncingWall() { return m_activateBouncingWall; }
            void setCyclicWall(bool Cyclicwall) { m_activateCyclicWall = Cyclicwall; }
            float getCyclicWall() { return m_activateCyclicWall; }
            void setmaxVelocity(float maxVelocity) { m_maxVelocity = maxVelocity; }
            float getmaxVelocity() { return m_maxVelocity; }
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

            void updateParticle(Entity& particle);
            void bouncingWall(Entity& particle);
            void cyclicWall(Entity& particle); // WIP
            void randomWall(Entity& particle); // WIP
            float m_maxVelocity;           
            bool m_activateBouncingWall;     
            bool m_activateCyclicWall;       
            bool m_forcedmaxspeed;
            bool m_pause;

            
    };
}

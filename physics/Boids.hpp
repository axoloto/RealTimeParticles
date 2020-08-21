#pragma once 

#include<array>
#include"Physics.hpp"

namespace Core {

    class Boids : public Physics {
        public:
            Boids(int boxSize, int numEntities);
            ~Boids() = default;

            void updatePhysics() override;
            void setmaxVelocity(float maxVelocity) { m_maxVelocity = maxVelocity; }
            float getmaxVelocity() { return m_maxVelocity; }
            void setbouncingWall(bool bouncingwall) { m_bouncingwall = bouncingwall; }
            float getbouncingWall() { return m_bouncingwall; }

        private:
            void generateBoids();
            void updateBoid(Entity& boid);
            void bouncingWall(Entity& boid);
            float m_maxVelocity;
            bool m_bouncingwall;
    };
}



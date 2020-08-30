
#pragma once

#include<memory>
#include "Physics.hpp"


namespace UI {

    class PhysicsWidget {
        public:
        explicit PhysicsWidget(Core::Physics &physicsEngine ) : m_physicsEngine(physicsEngine) {};
        virtual ~PhysicsWidget() = default;

        virtual void display() = 0;

        protected:
            Core::Physics &m_physicsEngine;
    };

    class BoidsWidget : public PhysicsWidget {
        public:
        explicit BoidsWidget(Core::Physics &physicsEngine ) : PhysicsWidget(physicsEngine) {};
        ~BoidsWidget() override = default;
        void display() override;

        private:

    };
}
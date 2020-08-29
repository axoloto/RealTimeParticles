
#pragma once

#include<memory>
#include "Physics.hpp"


namespace UI {

    class PhysicsWidget {
        public:
        explicit PhysicsWidget(const Core::Physics &physicsEngine ) : m_physicsEngine(physicsEngine) {};
        virtual ~PhysicsWidget() = default;

        virtual void display() = 0;

        protected:
            const Core::Physics &m_physicsEngine;
    };

    class BoidsWidget : public PhysicsWidget {
        public:
        explicit BoidsWidget(const Core::Physics &physicsEngine ) : PhysicsWidget(physicsEngine) {};
        ~BoidsWidget() override = default;
        void display() override;

        private:

    };
}
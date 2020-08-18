
#pragma once

#include "imgui/imgui.h"
#include "diligentGraphics/Math.hpp"
#include <SDL.h>
#include "OGLRender.hpp"
#include "Physics.hpp"
#include "PhysicsWidget.hpp"

class ParticleSystemApp {
        public:
            ParticleSystemApp();
            ~ParticleSystemApp() = default;
            void run();
            bool isInit() const { return m_init; }

        private:
            bool initWindow();
            bool closeWindow();
            bool checkSDLStatus();
            void checkMouseState();
            void displayMainWidget();

            std::shared_ptr<Core::Physics> m_physicsEngine;
            std::unique_ptr<Render::OGLRender> m_OGLRender;
            std::unique_ptr<UI::PhysicsWidget> m_physicsWidget;

            SDL_Window* m_window;
            SDL_GLContext m_OGLContext;

            Math::int2 m_windowSize;
            Math::int2 m_mousePrevPos;
            ImVec4 m_backGroundColor;
            bool m_buttonLeftActivated;
            bool m_buttonRightActivated;
            bool m_init;

            int m_numEntities;
            int m_boxSize;
};

#pragma once

#include "imgui/imgui.h"
#include "diligentGraphics/Math.hpp"
#include <SDL.h>
#include "OGLRender.hpp"

class BoidsApp {
        public:
            BoidsApp();
            ~BoidsApp() = default;
            void run();
            
        private:
            bool initOGL();
            bool closeOGL();
            bool checkSDLStatus();
            void checkMouseState();

            std::unique_ptr<Render::OGLRender> m_OGLRender;

            SDL_Window* m_window;
            SDL_GLContext m_OGLContext;

            Math::int2 m_mousePrevPos;
            ImVec4 m_backGroundColor;
            bool m_buttonLeftActivated;
            bool m_buttonRightActivated;
};
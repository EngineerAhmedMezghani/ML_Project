<template>
  <div class="min-h-screen flex">
    <!-- Sidebar -->
    <aside class="w-64 bg-white border-r border-gray-200 fixed h-full">
      <div class="p-6">
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
            <Brain class="w-5 h-5 text-white" />
          </div>
          <h1 class="text-xl font-bold text-gray-900">ChurnPredict</h1>
        </div>
        <p class="text-xs text-gray-500 mt-1 ml-10">ML Customer Analytics</p>
      </div>
      
      <nav class="px-4 py-4 space-y-1">
        <router-link to="/" class="nav-link" :class="{ active: $route.path === '/' }">
          <LayoutDashboard class="w-5 h-5" />
          Dashboard
        </router-link>
        <router-link to="/predict" class="nav-link" :class="{ active: $route.path === '/predict' }">
          <Sparkles class="w-5 h-5" />
          Predict
        </router-link>
        <router-link to="/history" class="nav-link" :class="{ active: $route.path === '/history' }">
          <HistoryIcon class="w-5 h-5" />
          History
        </router-link>
      </nav>

      <div class="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
        <div class="flex items-center gap-3 px-4 py-2">
          <div class="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
            <User class="w-4 h-4 text-gray-600" />
          </div>
          <div>
            <p class="text-sm font-medium text-gray-900">Admin User</p>
            <p class="text-xs text-gray-500">admin@company.com</p>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 ml-64">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { Brain, LayoutDashboard, Sparkles, History as HistoryIcon, User } from 'lucide-vue-next'
</script>

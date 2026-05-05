<template>
  <div class="p-8">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
      <p class="text-gray-600 mt-1">
        Overview of customer churn predictions and system status.
      </p>
    </div>

    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <div class="card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">Total Predictions</p>
            <p class="text-3xl font-bold text-gray-900">{{ stats.total_predictions }}</p>
          </div>
          <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
            <Activity class="w-6 h-6 text-blue-600" />
          </div>
        </div>
      </div>

      <div class="card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">Churn Risk</p>
            <p class="text-3xl font-bold text-red-600">{{ stats.churn_count }}</p>
          </div>
          <div class="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
            <AlertTriangle class="w-6 h-6 text-red-600" />
          </div>
        </div>
      </div>

      <div class="card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">Retention</p>
            <p class="text-3xl font-bold text-green-600">{{ stats.retention_count }}</p>
          </div>
          <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
            <CheckCircle class="w-6 h-6 text-green-600" />
          </div>
        </div>
      </div>

      <div class="card">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600">Avg Confidence</p>
            <p class="text-3xl font-bold text-primary-600">{{ stats.avg_confidence }}%</p>
          </div>
          <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
            <TrendingUp class="w-6 h-6 text-purple-600" />
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div class="card">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div class="space-y-3">
          <router-link to="/predict" class="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
            <div class="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
              <Sparkles class="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <p class="font-medium text-gray-900">New Prediction</p>
              <p class="text-sm text-gray-500">Make a new churn prediction</p>
            </div>
            <ChevronRight class="w-5 h-5 text-gray-400 ml-auto" />
          </router-link>

          <router-link to="/history" class="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
            <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
              <History class="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p class="font-medium text-gray-900">View History</p>
              <p class="text-sm text-gray-500">See past predictions</p>
            </div>
            <ChevronRight class="w-5 h-5 text-gray-400 ml-auto" />
          </router-link>
        </div>
      </div>

      <div class="card">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">System Status</h3>
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-gray-600">API Status</span>
            <span class="flex items-center gap-2">
              <span class="w-2 h-2 bg-green-500 rounded-full"></span>
              <span class="text-sm font-medium text-green-600">Online</span>
            </span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-gray-600">Model Status</span>
            <span class="flex items-center gap-2">
              <span class="w-2 h-2 bg-green-500 rounded-full"></span>
              <span class="text-sm font-medium text-green-600">Loaded</span>
            </span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-gray-600">Last Updated</span>
            <span class="text-sm text-gray-900">{{ lastUpdated }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Activity, AlertTriangle, CheckCircle, TrendingUp, Sparkles, History, ChevronRight } from 'lucide-vue-next'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const stats = ref({
  total_predictions: 0,
  churn_count: 0,
  retention_count: 0,
  avg_confidence: 0,
})

const lastUpdated = ref('Never')

onMounted(async () => {
  await fetchStats()
})

async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/history`)
    if (res.ok) {
      const data = await res.json()
      const history = data.history || []
      
      stats.value.total_predictions = history.length
      
      let churnCount = 0
      let confidenceSum = 0
      
      history.forEach(item => {
        if (item.prediction === 1 || item.churn?.prediction === 1) {
          churnCount++
        }
        if (item.churn?.probability) {
          confidenceSum += item.churn.probability
        }
      })
      
      stats.value.churn_count = churnCount
      stats.value.retention_count = history.length - churnCount
      stats.value.avg_confidence = history.length > 0 
        ? Math.round((confidenceSum / history.length) * 100) 
        : 0
      
      lastUpdated.value = new Date().toLocaleString()
    }
  } catch (e) {
    console.error('Failed to fetch stats:', e)
  }
}
</script>

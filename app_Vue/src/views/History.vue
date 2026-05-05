<template>
  <div class="p-8">
    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Prediction History</h1>
        <p class="text-gray-600 mt-1">View and manage all churn predictions</p>
      </div>
      <div class="flex gap-3">
        <button class="btn-secondary flex items-center gap-2">
          <Download class="w-4 h-4" />
          Export CSV
        </button>
        <button class="btn-secondary flex items-center gap-2">
          <Filter class="w-4 h-4" />
          Filter
        </button>
      </div>
    </div>

    <!-- Filters -->
    <div class="card mb-6">
      <div class="flex flex-wrap gap-4">
        <div class="flex-1 min-w-[200px]">
          <label class="block text-xs font-medium text-gray-700 mb-1">Search</label>
          <div class="relative">
            <Search class="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input 
              v-model="searchQuery" 
              type="text" 
              class="input-field pl-10" 
              placeholder="Search by customer ID or name..."
            />
          </div>
        </div>
        <div class="w-40">
          <label class="block text-xs font-medium text-gray-700 mb-1">Risk Level</label>
          <select v-model="riskFilter" class="input-field">
            <option value="">All</option>
            <option value="High">High</option>
            <option value="Medium">Medium</option>
            <option value="Low">Low</option>
          </select>
        </div>
        <div class="w-40">
          <label class="block text-xs font-medium text-gray-700 mb-1">Date Range</label>
          <select v-model="dateFilter" class="input-field">
            <option value="">All Time</option>
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Results Table -->
    <div class="card overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full">
          <thead class="bg-gray-50">
            <tr>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">
                <input type="checkbox" class="rounded border-gray-300" />
              </th>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">Customer</th>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">Risk Level</th>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">Probability</th>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">Prediction</th>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">Date</th>
              <th class="text-left py-3 px-4 text-xs font-medium text-gray-600 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-200">
            <tr v-for="item in filteredHistory" :key="item.id" class="hover:bg-gray-50">
              <td class="py-4 px-4">
                <input type="checkbox" class="rounded border-gray-300" />
              </td>
              <td class="py-4 px-4">
                <div class="flex items-center">
                  <div class="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center mr-3">
                    <User class="w-4 h-4 text-gray-500" />
                  </div>
                  <div>
                    <p class="text-sm font-medium text-gray-900">{{ item.name }}</p>
                    <p class="text-xs text-gray-500">{{ item.customerId }}</p>
                  </div>
                </div>
              </td>
              <td class="py-4 px-4">
                <span :class="getRiskBadgeClass(item.riskLevel)">
                  {{ item.riskLevel }}
                </span>
              </td>
              <td class="py-4 px-4">
                <div class="flex items-center gap-2">
                  <div class="w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      class="h-2 rounded-full" 
                      :class="getProbabilityBarClass(item.probability)"
                      :style="{ width: (item.probability * 100) + '%' }"
                    ></div>
                  </div>
                  <span class="text-sm text-gray-900">{{ (item.probability * 100).toFixed(1) }}%</span>
                </div>
              </td>
              <td class="py-4 px-4">
                <span :class="item.prediction === 1 ? 'text-red-600' : 'text-green-600'" class="text-sm font-medium">
                  {{ item.prediction === 1 ? 'Churn' : 'Loyal' }}
                </span>
              </td>
              <td class="py-4 px-4 text-sm text-gray-600">
                {{ item.date }}
              </td>
              <td class="py-4 px-4">
                <div class="flex items-center gap-2">
                  <button class="p-1 hover:bg-gray-100 rounded" title="View Details">
                    <Eye class="w-4 h-4 text-gray-500" />
                  </button>
                  <button class="p-1 hover:bg-gray-100 rounded" title="Download Report">
                    <FileText class="w-4 h-4 text-gray-500" />
                  </button>
                  <button class="p-1 hover:bg-gray-100 rounded" title="Delete">
                    <Trash2 class="w-4 h-4 text-gray-500" />
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Pagination -->
      <div class="flex items-center justify-between px-4 py-3 border-t border-gray-200">
        <div class="text-sm text-gray-600">
          Showing {{ (currentPage - 1) * itemsPerPage + 1 }} to {{ Math.min(currentPage * itemsPerPage, totalItems) }} of {{ totalItems }} results
        </div>
        <div class="flex items-center gap-2">
          <button 
            @click="currentPage--" 
            :disabled="currentPage === 1"
            class="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 hover:bg-gray-50"
          >
            Previous
          </button>
          <span class="text-sm text-gray-600">Page {{ currentPage }} of {{ totalPages }}</span>
          <button 
            @click="currentPage++" 
            :disabled="currentPage === totalPages"
            class="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { Search, Download, Filter, User, Eye, FileText, Trash2 } from 'lucide-vue-next'

const searchQuery = ref('')
const riskFilter = ref('')
const dateFilter = ref('')
const currentPage = ref(1)
const itemsPerPage = 10

// Mock data - replace with actual API data
const historyData = ref([
  { id: 1, customerId: 'CUST-14300', name: 'John Smith', riskLevel: 'Low', probability: 0.15, prediction: 0, date: '2024-06-15 14:30' },
  { id: 2, customerId: 'CUST-17848', name: 'Sarah Johnson', riskLevel: 'Medium', probability: 0.45, prediction: 0, date: '2024-06-15 13:15' },
  { id: 3, customerId: 'CUST-14873', name: 'Mike Brown', riskLevel: 'High', probability: 0.85, prediction: 1, date: '2024-06-14 16:45' },
  { id: 4, customerId: 'CUST-15067', name: 'Emily Davis', riskLevel: 'Low', probability: 0.22, prediction: 0, date: '2024-06-14 11:20' },
  { id: 5, customerId: 'CUST-17722', name: 'Chris Wilson', riskLevel: 'High', probability: 0.78, prediction: 1, date: '2024-06-13 09:00' },
  { id: 6, customerId: 'CUST-12410', name: 'Lisa Anderson', riskLevel: 'Medium', probability: 0.52, prediction: 1, date: '2024-06-13 15:30' },
  { id: 7, customerId: 'CUST-12401', name: 'David Martinez', riskLevel: 'Low', probability: 0.18, prediction: 0, date: '2024-06-12 10:45' },
  { id: 8, customerId: 'CUST-14755', name: 'Jennifer Lee', riskLevel: 'High', probability: 0.92, prediction: 1, date: '2024-06-12 14:15' },
  { id: 9, customerId: 'CUST-13727', name: 'Robert Taylor', riskLevel: 'Medium', probability: 0.38, prediction: 0, date: '2024-06-11 16:00' },
  { id: 10, customerId: 'CUST-14301', name: 'Amanda White', riskLevel: 'Low', probability: 0.12, prediction: 0, date: '2024-06-11 11:30' },
  { id: 11, customerId: 'CUST-17849', name: 'James Thompson', riskLevel: 'High', probability: 0.73, prediction: 1, date: '2024-06-10 13:45' },
  { id: 12, customerId: 'CUST-14874', name: 'Michelle Garcia', riskLevel: 'Medium', probability: 0.48, prediction: 0, date: '2024-06-10 09:15' }
])

const filteredHistory = computed(() => {
  let result = historyData.value

  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(item => 
      item.customerId.toLowerCase().includes(query) ||
      item.name.toLowerCase().includes(query)
    )
  }

  if (riskFilter.value) {
    result = result.filter(item => item.riskLevel === riskFilter.value)
  }

  // Pagination
  const start = (currentPage.value - 1) * itemsPerPage
  const end = start + itemsPerPage
  return result.slice(start, end)
})

const totalItems = computed(() => historyData.value.length)
const totalPages = computed(() => Math.ceil(totalItems.value / itemsPerPage))

const getRiskBadgeClass = (risk) => {
  const classes = {
    'Low': 'px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-700',
    'Medium': 'px-2 py-1 text-xs font-medium rounded-full bg-yellow-100 text-yellow-700',
    'High': 'px-2 py-1 text-xs font-medium rounded-full bg-red-100 text-red-700'
  }
  return classes[risk] || classes['Low']
}

const getProbabilityBarClass = (prob) => {
  if (prob > 0.7) return 'bg-red-500'
  if (prob > 0.3) return 'bg-yellow-500'
  return 'bg-green-500'
}
</script>

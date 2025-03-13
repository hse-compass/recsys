<template>
    <v-container>
      <v-text-field
        v-model="studentId"
        label="Введите ID студента"
        type="number"
        clearable
      ></v-text-field>
  
      <v-btn @click="getRecommendations" :loading="loading" color="primary" class="mt-2">Получить рекомендации</v-btn>
  
      <v-alert v-if="error" type="error" class="mt-3">{{ error }}</v-alert>
  
      <v-list v-if="recommendations.length" class="mt-3">
        <v-list-item v-for="rec in recommendations" :key="rec.id">
          <v-list-item-title>{{ rec.name }}</v-list-item-title>
          <v-list-item-subtitle>
            Факультет: {{ rec.faculty }}, Степень образования: {{ rec.level_of_education }}, Интересы {{ rec.interests }}, Языки: {{ rec.languages }}, Возраст: {{ rec.age }}, Комната: {{ rec.room_number }}, Общежитие: {{ rec.hostel_id }}
          </v-list-item-subtitle>
        </v-list-item>
      </v-list>
    </v-container>
  </template>
  
  <script setup lang="ts">
  import { ref } from 'vue'
  import axios from 'axios'
  
  const studentId = ref<number | null>(null)
  const recommendations = ref<any[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  const getRecommendations = async () => {
    if (!studentId.value) {
      error.value = 'Введите ID студента'
      return
    }
  
    loading.value = true
    error.value = null
  
    try {
      const response = await axios.get(`http://127.0.0.1:8000/recommendations/${studentId.value}`)
      recommendations.value = response.data
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Произошла ошибка'
    } finally {
      loading.value = false
    }
  }
  </script>
  
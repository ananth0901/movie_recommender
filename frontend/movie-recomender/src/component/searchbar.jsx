import React from 'react'

export const searchbar = () => {
  return (
    <div class="mx-auto mt-5 w-screen max-w-screen-md py-20 leading-6"> 
  <form class="relative mx-auto flex w-full max-w-2xl items-center justify-between rounded-md border shadow-lg"> 
    <svg class="absolute left-2 block h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="8" class=""></circle>
      <line x1="21" y1="21" x2="16.65" y2="16.65" class=""></line>
    </svg>
    <input type="name" name="search" class="h-14 w-full rounded-md py-4 pr-40 pl-12 outline-none focus:ring-2" placeholder="City, Address, Zip :" />
    <button type="submit" class="absolute right-0 mr-1 inline-flex h-12 items-center justify-center rounded-lg bg-gray-900 px-10 font-medium text-white focus:ring-4 hover:bg-gray-700">Search</button>
  </form>
</div>


  )
}
